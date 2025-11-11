using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;


class Program
{
    static readonly string dataPath = "data/fake.csv";
    static readonly string modelPath = "models/HoaxModel.zip";

    static void Main()
    {
        var mlContext = new MLContext(seed: 42);

        if (File.Exists(modelPath))
        {
            Console.WriteLine("📦 Memuat model yang sudah ada...");
            var loadedModel = mlContext.Model.Load(modelPath, out _);

            RunPrediction(mlContext, loadedModel);
            return;
        }

        Console.WriteLine("🧠 Melatih model baru...");
        IDataView dataView = mlContext.Data.LoadFromTextFile<NewsData>(
            path: dataPath, hasHeader: true, separatorChar: ',', allowQuoting: true);

        var sampleData = mlContext.Data.TakeRows(dataView, 1000); // cuma 1000 baris
        var split = mlContext.Data.TrainTestSplit(sampleData, testFraction: 0.2);


        var pipeline = mlContext.Transforms.Conversion.MapValueToKey("LabelKey", "Label")
            .Append(mlContext.Transforms.Text.FeaturizeText("JudulFeats", "Judul"))
            .Append(mlContext.Transforms.Text.FeaturizeText("NarasiFeats", "Narasi"))
            .Append(mlContext.Transforms.Concatenate("Features", "JudulFeats", "NarasiFeats"))
            .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("LabelKey", "Features"))
            .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedLabel"));

        var model = pipeline.Fit(split.TrainSet);

        var predictions = model.Transform(split.TestSet);
         var metrics = mlContext.MulticlassClassification.Evaluate(
            predictions,
            labelColumnName: "LabelKey",
            scoreColumnName: "Score",
            predictedLabelColumnName: "PredictedLabel");

        Console.WriteLine($"MicroAccuracy: {metrics.MicroAccuracy:P2}");
        Console.WriteLine($"MacroAccuracy: {metrics.MacroAccuracy:P2}");

        if (!Directory.Exists("models"))
        {
            Directory.CreateDirectory("models");
        }


        mlContext.Model.Save(model, dataView.Schema, modelPath);
        Console.WriteLine($"✅ Model disimpan ke: {modelPath}");

        RunPrediction(mlContext, model);
    }

    static void RunPrediction(MLContext mlContext, ITransformer model)
    {
        Console.WriteLine("\nMasukkan judul berita:");
        string? judul = Console.ReadLine();

        Console.WriteLine("\nMasukkan isi/narasi berita:");
        string? narasi = Console.ReadLine();

        var predEngine = mlContext.Model.CreatePredictionEngine<NewsData, NewsPrediction>(model);

        var sample = new NewsData { Judul = judul, Narasi = narasi };
        var prediction = predEngine.Predict(sample);

        Console.WriteLine("\n=== HASIL PREDIKSI ===");
        Console.WriteLine($"Label: {prediction.PredictedLabel}");
        if (prediction.Score != null)
            Console.WriteLine($"Confidence: {prediction.Score.Max():P2}");
        Console.WriteLine("=======================");
    }

} 
