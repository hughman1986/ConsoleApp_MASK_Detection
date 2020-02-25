using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Vision;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using static Microsoft.ML.Transforms.ValueToKeyMappingEstimator;



namespace ConsoleApp_MASK_Detection
{
    class Program
    {
        static void Main(string[] args)
        {
            //03 準備環境與 load 檔案 
            string dataPath = Path.Combine(Environment.CurrentDirectory, @"../../../data");
            string imageFolderPath = Path.Combine(dataPath, "mask_dataset");

            var mlContext = new MLContext(seed: 1); // 

            IEnumerable<ImageData> images = LoadImagesFromDirectory( // 讀出檔案路徑清單 
                folder: imageFolderPath,
                useFolderNameAsLabel: true);
            IDataView fullImagesDataset = mlContext.Data.LoadFromEnumerable(images); //載入影像 
            IDataView shuffledFullImageFilePathsDataset = mlContext.Data.ShuffleRows(fullImagesDataset);

            //04 數據前處理 
            IDataView shuffledFullImagesDataset = mlContext.Transforms.Conversion.
                MapValueToKey(outputColumnName: "LabelAsKey", inputColumnName: "Label", keyOrdinality: KeyOrdinality.ByValue) // label 轉成 index key 
                .Append(mlContext.Transforms.LoadRawImageBytes(
                outputColumnName: "Image",
                imageFolder: imageFolderPath,
                inputColumnName: "ImagePath"))
                .Fit(shuffledFullImageFilePathsDataset)
                .Transform(shuffledFullImageFilePathsDataset);

            //05 切割資料 
            var trainTestData = mlContext.Data.TrainTestSplit(shuffledFullImagesDataset, testFraction: 0.2);
            IDataView trainDataView = trainTestData.TrainSet;
            IDataView testDataView = trainTestData.TestSet;


            //06 Train model
            var options = new ImageClassificationTrainer.Options()
            {
                FeatureColumnName = "Image",
                LabelColumnName = "LabelAsKey",
                Arch = ImageClassificationTrainer.Architecture.MobilenetV2, //選用的模型, 會直接上網下載 
                Epoch = 50,
                BatchSize = 10,
                LearningRate = 0.01f,
                MetricsCallback = (metrics) => Console.WriteLine(metrics),
                ValidationSet = testDataView
            };

            var pipeline = mlContext.MulticlassClassification.Trainers.ImageClassification(options)
                            .Append(mlContext.Transforms.Conversion.MapKeyToValue(
                                outputColumnName: "PredictedLabel",
                                inputColumnName: "PredictedLabel"));

            ITransformer trainedModel = pipeline.Fit(trainDataView);

            var predictionsDataView = trainedModel.Transform(testDataView);


            //07 Evaluate
            var metrics = mlContext.MulticlassClassification.Evaluate(predictionsDataView,
                labelColumnName: "LabelAsKey", predictedLabelColumnName: "PredictedLabel");

            Console.WriteLine($"************************************************************");
            Console.WriteLine($"* Metrics for TensorFlow DNN Transfer Learning multi-class classification model ");
            Console.WriteLine($"*-----------------------------------------------------------");
            Console.WriteLine($" AccuracyMacro = {metrics.MacroAccuracy:0.####}, a value between 0 and 1, the closer to 1, the better");
            Console.WriteLine($" AccuracyMicro = {metrics.MicroAccuracy:0.####}, a value between 0 and 1, the closer to 1, the better");
            Console.WriteLine($" LogLoss = {metrics.LogLoss:0.####}, the closer to 0, the better");

            int i = 0;
            foreach (var classLogLoss in metrics.PerClassLogLoss)
            {
                i++;
                Console.WriteLine($" LogLoss for class {i} = {classLogLoss:0.####}, the closer to 0, the better");
            }
            Console.WriteLine($"************************************************************");


            //08 做預測 
            var predictionEngine = mlContext.Model
                .CreatePredictionEngine<InMemoryImageData, ImagePrediction>(trainedModel);

            List<InMemoryImageData> imagesToPredict = new List<InMemoryImageData>();

            var images_path = Directory.GetFiles(Path.Combine(dataPath, "mask_dataset"));

            foreach (var filepath in images_path)
            {
                var imageToPredict = new InMemoryImageData(File.ReadAllBytes(filepath),
                    "mask", Path.GetFileName(filepath));
                imagesToPredict.Add(imageToPredict);
            }

            foreach (var img2pre in imagesToPredict)
            {
                var prediction = predictionEngine.Predict(img2pre);

                Console.WriteLine(
                $"Image Filename : [{img2pre.ImageFileName}], " +
                $"Scores : [{string.Join(",", prediction.Score)}], " +
                $"Predicted Label : {prediction.PredictedLabel}");
            }

            new System.Threading.AutoResetEvent(false).WaitOne();
        }





        /// <summary>
        /// 讀出檔案路徑清單
        /// 載入檔案, 過濾檔名, 回傳ImageData 列表 
        /// </summary>
        /// <param name="folder"></param>
        /// <param name="useFolderNameAsLabel"></param>
        /// <returns></returns>
        public static IEnumerable<ImageData> LoadImagesFromDirectory(string folder, bool useFolderNameAsLabel = true)
        {
            var imagesPath = Directory
            .GetFiles(folder, "*", searchOption: SearchOption.AllDirectories)
            .Where(x => Path.GetExtension(x) == ".jpg" || Path.GetExtension(x) == ".png");

            var result = useFolderNameAsLabel
            ? imagesPath.Select(imagePath => (imagePath, Directory.GetParent(imagePath).Name))
            : imagesPath.Select(imagePath =>
            {
                var label = Path.GetFileName(imagePath);
                for (var index = 0; index < label.Length; index++)
                {
                    if (!char.IsLetter(label[index]))
                    {
                        label = label.Substring(0, index);
                        break;
                    }
                }
                return (imagePath, label); // 回傳路徑與 label 
            });

            return result.Select(x => new ImageData(x.imagePath, x.Item2));
        }



        // ## 定義input與output資料結構: 

        /// <summary>
        /// 圖片的格式與名稱 
        /// </summary>
        public class ImageData
        {
            public ImageData(string imagePath, string label)
            {
                ImagePath = imagePath;
                Label = label;
            }

            public readonly string ImagePath;
            public readonly string Label;
        }


        /// <summary>
        /// 檔案載入在memory內的資料結構 
        /// </summary>
        public class InMemoryImageData
        {
            public InMemoryImageData(byte[] image, string label, string imageFileName)
            {
                Image = image;
                Label = label;
                ImageFileName = imageFileName;
            }

            public readonly byte[] Image;
            public readonly string Label;
            public readonly string ImageFileName;
        }

        /// <summary>
        /// 預測結果輸出格式 
        /// </summary>
        public class ImagePrediction
        {
            [ColumnName("Score")]
            public float[] Score;

            [ColumnName("PredictedLabel")]
            public string PredictedLabel;
        }


    }
}
