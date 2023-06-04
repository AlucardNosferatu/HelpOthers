using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Windows.Forms.DataVisualization.Charting;

namespace WindowsFormsApp1
{
    public partial class Form1 : Form
    {
        public Form1(double[] score,String[] text)
        {
            InitializeComponent();
            chart1.Series[0].Points.DataBindY(score);
            richTextBox1.Text = "";
            foreach(String S in text)
            {
                richTextBox1.Text += S;
                richTextBox1.Text += "\n";
            }
            richTextBox1.SelectAll();
            Font font = new Font(FontFamily.GenericMonospace, 8, FontStyle.Regular);
            this.richTextBox1.SelectionFont = font;
        }

        private void Chart1_Click(object sender, EventArgs e)
        {

        }

        private void WebBrowser1_DocumentCompleted(object sender, WebBrowserDocumentCompletedEventArgs e)
        {

        }

        private void Button1_Click(object sender, EventArgs e)
        {

                
        }
    }
}
