using System;
using System.Collections;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Windows.Forms.DataVisualization.Charting;
using MP3Sharp;
using DSP;


namespace WindowsFormsApp1
{
    public partial class Form1 : Form
    {
        private readonly API api = new API();

        public Form1()
        {
            InitializeComponent();
        }
        private static double[] LoadMP3AsArray(String fileName = "Music\\Sample.mp3")
        {
            MP3Stream mp3 = new MP3Stream(fileName:fileName);
            byte[] buffer = new byte[4096];
            int bytesReturned = 1;
            int totalBytesRead = 0;
            ArrayList wav = new ArrayList();
            int Sum;
            #region:obsoleted full-sized extraction
            //byte[] wav = new byte[0];
            //byte[] wav_new;
            #endregion
            while (bytesReturned > 0)
            {
                Sum = 0x00;
                bytesReturned = mp3.Read(buffer, 0, buffer.Length);
                totalBytesRead += bytesReturned;
                foreach (byte b in buffer)
                {
                    Sum += (int)b;
                }
                wav.Add(Sum / buffer.Length);
                #region:obsoleted full-sized extraction
                //wav_new = new byte[wav.Length + buffer.Length];
                //wav.CopyTo(wav_new, 0);
                //buffer.CopyTo(wav_new, wav.Length);
                //wav = wav_new;
                #endregion
            }
            mp3.Close();
            int[] wav_array_int = (int[])wav.ToArray(typeof(int));
            double[] wav_array = new double[wav_array_int.Length];
            for (int i = 0; i < wav_array_int.Length; i++)
            {
                wav_array[i] = (double)wav_array_int[i];
            }
            return wav_array;
        }
        private void Button1_Click(object sender, EventArgs e)
        {
            #region:Analyze MP3
            double[] wav = LoadMP3AsArray();
            int TargetLength = (int)FourierTransform.NextPowerOfTwo((uint)wav.Length);
            double[] wav_new = new double[TargetLength];
            wav.CopyTo(wav_new, 0);
            for(int i = wav.Length; i < TargetLength; i++)
            {
                wav_new[i] = wav[wav.Length - 1];
            }
            double[] fs = FourierTransform.Spectrum(ref wav_new, method: 1);
            double[] fs_dB = Amp2dB(fs);
            chart1.Series[0].Points.DataBindY(fs_dB);
            chart1.Series[0].ChartType = SeriesChartType.Spline;
            #endregion
        }
        private static double[] Amp2dB(double[] Amp)
        {
            double[] dB = new double[Amp.Length];
            for(int i = 0; i < dB.Length; i++)
            {
                dB[i] = 20 * Math.Log10(Amp[i] / Amp.Max());
            }
            return dB;
        }
        private static double[] subArray(double[] Input)
        {
            return (double[])(new ArrayList(Input).GetRange(0, 100).ToArray(typeof(double)));
        }
        private async void Button2_Click(object sender, EventArgs e)
        {
            #region:Get MP3
            string url = textBox1.Text;
            if (url.Contains("song?id=") | url.Contains("song/"))
            {
                Regex re = new Regex(@"(song\?id=)\d+", RegexOptions.Compiled);
                string id = re.Match(url).ToString();
                if (id != "")
                {
                    id = id.Replace("song?id=", "");
                }
                else
                {
                    re = new Regex(@"(song/)\d+", RegexOptions.Compiled);
                    id = re.Match(url).ToString();
                    id = id.Replace("song/", "");
                }
                string name = "";
                name = await api.GetSingle(id);
            }
            string appPath = Application.StartupPath;
            await api.DownloadAll(appPath);
            textBox1.Text = "Done.";
            #endregion
        }
    }
}
