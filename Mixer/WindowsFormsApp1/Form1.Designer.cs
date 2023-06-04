namespace WindowsFormsApp1
{
    partial class Form1
    {
        /// <summary>
        /// 必需的设计器变量。
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// 清理所有正在使用的资源。
        /// </summary>
        /// <param name="disposing">如果应释放托管资源，为 true；否则为 false。</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows 窗体设计器生成的代码

        /// <summary>
        /// 设计器支持所需的方法 - 不要修改
        /// 使用代码编辑器修改此方法的内容。
        /// </summary>
        private void InitializeComponent()
        {
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(Form1));
            this.textBox1 = new System.Windows.Forms.TextBox();
            this.WMP = new AxWMPLib.AxWindowsMediaPlayer();
            this.WMP2 = new AxWMPLib.AxWindowsMediaPlayer();
            this.WMP3 = new AxWMPLib.AxWindowsMediaPlayer();
            this.WMP4 = new AxWMPLib.AxWindowsMediaPlayer();
            this.WMP5 = new AxWMPLib.AxWindowsMediaPlayer();
            ((System.ComponentModel.ISupportInitialize)(this.WMP)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.WMP2)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.WMP3)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.WMP4)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.WMP5)).BeginInit();
            this.SuspendLayout();
            // 
            // textBox1
            // 
            this.textBox1.Location = new System.Drawing.Point(13, 13);
            this.textBox1.Name = "textBox1";
            this.textBox1.ReadOnly = true;
            this.textBox1.Size = new System.Drawing.Size(983, 21);
            this.textBox1.TabIndex = 3;
            // 
            // WMP
            // 
            this.WMP.Enabled = true;
            this.WMP.Location = new System.Drawing.Point(13, 41);
            this.WMP.Name = "WMP";
            this.WMP.OcxState = ((System.Windows.Forms.AxHost.State)(resources.GetObject("WMP.OcxState")));
            this.WMP.Size = new System.Drawing.Size(983, 58);
            this.WMP.TabIndex = 5;
            // 
            // WMP2
            // 
            this.WMP2.Enabled = true;
            this.WMP2.Location = new System.Drawing.Point(13, 114);
            this.WMP2.Name = "WMP2";
            this.WMP2.OcxState = ((System.Windows.Forms.AxHost.State)(resources.GetObject("WMP2.OcxState")));
            this.WMP2.Size = new System.Drawing.Size(983, 58);
            this.WMP2.TabIndex = 6;
            // 
            // WMP3
            // 
            this.WMP3.Enabled = true;
            this.WMP3.Location = new System.Drawing.Point(13, 187);
            this.WMP3.Name = "WMP3";
            this.WMP3.OcxState = ((System.Windows.Forms.AxHost.State)(resources.GetObject("WMP3.OcxState")));
            this.WMP3.Size = new System.Drawing.Size(983, 58);
            this.WMP3.TabIndex = 7;
            // 
            // WMP4
            // 
            this.WMP4.Enabled = true;
            this.WMP4.Location = new System.Drawing.Point(13, 260);
            this.WMP4.Name = "WMP4";
            this.WMP4.OcxState = ((System.Windows.Forms.AxHost.State)(resources.GetObject("WMP4.OcxState")));
            this.WMP4.Size = new System.Drawing.Size(983, 58);
            this.WMP4.TabIndex = 8;
            // 
            // WMP5
            // 
            this.WMP5.Enabled = true;
            this.WMP5.Location = new System.Drawing.Point(13, 333);
            this.WMP5.Name = "WMP5";
            this.WMP5.OcxState = ((System.Windows.Forms.AxHost.State)(resources.GetObject("WMP5.OcxState")));
            this.WMP5.Size = new System.Drawing.Size(983, 58);
            this.WMP5.TabIndex = 9;
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 12F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.AutoSize = true;
            this.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.ClientSize = new System.Drawing.Size(1008, 403);
            this.Controls.Add(this.WMP5);
            this.Controls.Add(this.WMP4);
            this.Controls.Add(this.WMP3);
            this.Controls.Add(this.WMP2);
            this.Controls.Add(this.WMP);
            this.Controls.Add(this.textBox1);
            this.KeyPreview = true;
            this.Name = "Form1";
            this.Text = "Form1";
            ((System.ComponentModel.ISupportInitialize)(this.WMP)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.WMP2)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.WMP3)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.WMP4)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.WMP5)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion
        private System.Windows.Forms.TextBox textBox1;
        private AxWMPLib.AxWindowsMediaPlayer WMP;
        private AxWMPLib.AxWindowsMediaPlayer WMP2;
        private AxWMPLib.AxWindowsMediaPlayer WMP3;
        private AxWMPLib.AxWindowsMediaPlayer WMP4;
        private AxWMPLib.AxWindowsMediaPlayer WMP5;
    }
}

