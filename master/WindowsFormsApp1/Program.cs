using System;
using System.Linq;
using System.Windows.Forms;
using Newtonsoft.Json.Linq;
using VaderSharp;
using System.Security.Cryptography;
using System.Text;
using System.Net.Http;
using System.Web;
using Newtonsoft.Json;

namespace WindowsFormsApp1
{
    static class Program
    {
        /// <summary>
        /// 应用程序的主入口点。
        /// </summary>
        [STAThread]
        static void Main()
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new Form1());
        }





        #region:NLP-Based
        private static String[] GetComments(int music_id)
        {
            NeteaseMusicAPI.NeteaseMusicAPI NMAPI = new NeteaseMusicAPI.NeteaseMusicAPI();
            int offset = 0;
            JObject Response = NMAPI.GetSongComments(id: music_id, offset: offset);
            String[] C = new string[(int)Response["total"]];
            JToken Comments;
            while (true)
            {
                Comments = Response["comments"];
                for (int i = offset; i < offset + Comments.Count(); i++)
                {
                    C[i] = (String)Comments[i - offset]["content"];
                }
                if (!(Boolean)Response["more"])
                {
                    break;
                }
                offset += 20;
                Response = NMAPI.GetSongComments(id: music_id, offset: offset);
            }
            return C;
        }
        private static String[] GetLyrics(int music_id)
        {
            NeteaseMusicAPI.NeteaseMusicAPI NMAPI = new NeteaseMusicAPI.NeteaseMusicAPI();
            JObject Music = NMAPI.GetLyric(id: music_id);
            String Lyric = (String)Music["lrc"]["lyric"];
            String[] Sentences = Lyric.Split('\n');
            return Sentences;
        }
        static double GetScore(String Input)
        {
            SentimentIntensityAnalyzer SIA = new SentimentIntensityAnalyzer();
            SentimentAnalysisResults SAR = SIA.PolarityScores(Input);
            double CS = SAR.Compound;
            return CS;
        }
        static double[] LoopTest(String[] Input)
        {
            double[] score = new double[Input.Length];
            for (int i = 0; i < Input.Length; i++)
            {
                String Temp;
                if (Input[i].Length > 10)
                {
                    Temp = Input[i].Substring(10);
                }
                else
                {
                    Temp = "";
                }
                double CS = GetScore(Temp);
                score[i] = CS;
            }
            return score;
        }
        static String GetTranslate(String Input)
        {
            String appid = "20190621000309477"; //你的appid
            String secretKey = "0PvcG1kzzqZbONCai3eR"; //你的密钥
            String myurl = "https://api.fanyi.baidu.com/api/trans/vip/translate";
            String q = Input;
            String fromLang = "zh";
            String toLang = "en";

            String salt = "1435660288";
            String sign = appid + q + salt + secretKey;
            
            MD5 m1 = MD5.Create();
            byte[] sign_bytes = m1.ComputeHash(Encoding.UTF8.GetBytes(sign));

            StringBuilder SB = new StringBuilder();
            for (int i = 0; i < sign_bytes.Length; i++)
            {
                SB.Append(sign_bytes[i].ToString("x2"));
            }

            myurl = myurl + "?appid=" + appid + "&q=" + HttpUtility.UrlEncode(q, Encoding.UTF8) + "&from=" + fromLang + "&to=" + toLang + "&salt=" + salt + "&sign=" + SB.ToString();

            HttpClient HC = new HttpClient();
            String Result = HC.GetStringAsync(myurl).Result;
            JToken ResultObj = (JToken)JsonConvert.DeserializeObject(Result);

            return (String)ResultObj["trans_result"][0]["dst"];
        }
        #endregion
    }

}
