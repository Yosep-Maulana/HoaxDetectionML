using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.IO;
using System.Linq;

public class NewsData
{
    [LoadColumn(0)] public string? Id { get; set; }              
    [LoadColumn(1)] public string? Label { get; set; }          
    [LoadColumn(2)] public string? Tanggal { get; set; }        
    [LoadColumn(3)] public string? Judul { get; set; }        
    [LoadColumn(4)] public string? Narasi { get; set; }          
    [LoadColumn(5)] public string? NamaFileGambar { get; set; }  
}

public class NewsPrediction
{
    [ColumnName("PredictedLabel")]
    public string? PredictedLabel { get; set; }

    public float[]? Score { get; set; }
}
