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
using System.Runtime.InteropServices;
using System.Reflection;
using System.Threading;

namespace WindowsFormsApp1
{
    public partial class Form1 : Form
    {
        #region:FSA-Based
        private readonly API api = new API();
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
        private async void MP3DownlaoderAsync(String url)
        {
            #region:Get MP3
            //string url = textBox1.Text;
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
            //textBox1.Text = "Done.";
            #endregion
        }
        private void MP3Analyzer()
        {
            #region:Analyze MP3
            double[] wav = LoadMP3AsArray();
            int TargetLength = (int)FourierTransform.NextPowerOfTwo((uint)wav.Length);
            double[] wav_new = new double[TargetLength];
            wav.CopyTo(wav_new, 0);
            for (int i = wav.Length; i < TargetLength; i++)
            {
                wav_new[i] = wav[wav.Length - 1];
            }
            double[] fs = FourierTransform.Spectrum(ref wav_new, method: 1);
            double[] fs_dB = Amp2dB(fs);
            //chart1.Series[0].Points.DataBindY(fs_dB);
            //chart1.Series[0].ChartType = SeriesChartType.Spline;
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
        #endregion


        #region:Mixer
        KeyboardHook kbh;
        Boolean[] KeyState;
        public delegate void PE(AxWMPLib.AxWindowsMediaPlayer WMP, String FileName);
        public PE pe;
        public delegate void SE(AxWMPLib.AxWindowsMediaPlayer WMP);
        public SE se;
        #endregion

        public Form1()
        {
            InitializeComponent();
            
        }
        #region:Mixer
        private void InitMixer()
        {
            System.Windows.Forms.Control.CheckForIllegalCrossThreadCalls = false;
            kbh = new KeyboardHook();
            kbh.KeyPressEvent += KP;
            kbh.KeyUpEvent += new KeyEventHandler(KU);
            kbh.Start();
            KeyState = new Boolean[3] { false, false, false };

            pe = new PE(PlayEvent);
            se = new SE(StopEvent);

            Thread CM = new Thread(new ThreadStart(delegate { ChannelManager(); }));
            CM.Start();
        }
        private void PlayEvent(AxWMPLib.AxWindowsMediaPlayer WMP,String FileName)
        {
            if (WMP.playState == WMPLib.WMPPlayState.wmppsUndefined)
            {
                WMP.URL = @FileName;
                WMP.Ctlcontrols.play();
            }
            else if (WMP.playState == WMPLib.WMPPlayState.wmppsStopped)
            {
                WMP.Ctlcontrols.play();
            }
        }
        private void StopEvent(AxWMPLib.AxWindowsMediaPlayer WMP)
        {
            if (WMP.playState == WMPLib.WMPPlayState.wmppsPlaying)
            {
                WMP.Ctlcontrols.stop();
            }
        }
        private void KP(object sender,KeyPressEventArgs e)
        {
            if (e.KeyChar == 'a')
            {
                KeyState[0] = true;
            }
            else if (e.KeyChar == 's')
            {
                KeyState[1] = true;
            }
            else if (e.KeyChar == 'd')
            {
                KeyState[2] = true;
            }
        }
        private void KU(object sender, KeyEventArgs e)
        {
            if (e.KeyValue == (int)Keys.A)
            {
                KeyState[0] = false;
            }
            else if (e.KeyValue == (int)Keys.S)
            {
                KeyState[1] = false;
            }
            else if (e.KeyValue == (int)Keys.D)
            {
                KeyState[2] = false;
            }
        }
        private void ChannelManager()
        {
            while (true)
            {
                textBox1.Text = KeyState[0].ToString() + " " + KeyState[1].ToString() + " " + KeyState[2].ToString();
                if (KeyState[0])
                {
                    //this.BeginInvoke(pe, WMP, "Music\\Sample.mp3");
                }
                else
                {
                    //this.BeginInvoke(se, WMP);
                }
                if (KeyState[1])
                {
                    //this.BeginInvoke(pe, WMP2, "Music\\Sample2.mp3");
                }
                else
                {
                    //this.BeginInvoke(se, WMP2);
                }
                if (KeyState[2])
                {
                    //this.BeginInvoke(pe, WMP3, "Music\\Sample3.mp3");
                }
                else
                {
                    //this.BeginInvoke(se, WMP3);
                }
            }
        }
    }
    class KeyboardHook
    {
        public event System.Windows.Forms.KeyEventHandler KeyDownEvent;
        public event KeyPressEventHandler KeyPressEvent;
        public event System.Windows.Forms.KeyEventHandler KeyUpEvent;
        public delegate int HookProc(int nCode, Int32 wParam, IntPtr lParam);
        static int hKeyboardHook = 0; //声明键盘钩子处理的初始值
        //值在Microsoft SDK的Winuser.h里查询
        // http://www.bianceng.cn/Programming/csharp/201410/45484.htm
        public const int WH_KEYBOARD_LL = 13;   //线程键盘钩子监听鼠标消息设为2，全局键盘监听鼠标消息设为13
        HookProc KeyboardHookProcedure; //声明KeyboardHookProcedure作为HookProc类型
        //键盘结构
        [StructLayout(LayoutKind.Sequential)]
        public class KeyboardHookStruct
        {
            public int vkCode;  //定一个虚拟键码。该代码必须有一个价值的范围1至254
            public int scanCode; // 指定的硬件扫描码的关键
            public int flags;  // 键标志
            public int time; // 指定的时间戳记的这个讯息
            public int dwExtraInfo; // 指定额外信息相关的信息
        }
        //使用此功能，安装了一个钩子
        [DllImport("user32.dll", CharSet = CharSet.Auto, CallingConvention = CallingConvention.StdCall)]
        public static extern int SetWindowsHookEx(int idHook, HookProc lpfn, IntPtr hInstance, int threadId);
        //调用此函数卸载钩子
        [DllImport("user32.dll", CharSet = CharSet.Auto, CallingConvention = CallingConvention.StdCall)]
        public static extern bool UnhookWindowsHookEx(int idHook);
        //使用此功能，通过信息钩子继续下一个钩子
        [DllImport("user32.dll", CharSet = CharSet.Auto, CallingConvention = CallingConvention.StdCall)]
        public static extern int CallNextHookEx(int idHook, int nCode, Int32 wParam, IntPtr lParam);
        // 取得当前线程编号（线程钩子需要用到）
        [DllImport("kernel32.dll")]
        static extern int GetCurrentThreadId();
        //使用WINDOWS API函数代替获取当前实例的函数,防止钩子失效
        [DllImport("kernel32.dll")]
        public static extern IntPtr GetModuleHandle(string name);
        public void Start()
        {
            // 安装键盘钩子
            if (hKeyboardHook == 0)
            {
                KeyboardHookProcedure = new HookProc(KeyboardHookProc);
                hKeyboardHook = SetWindowsHookEx(WH_KEYBOARD_LL, KeyboardHookProcedure, GetModuleHandle(System.Diagnostics.Process.GetCurrentProcess().MainModule.ModuleName), 0);
                //hKeyboardHook = SetWindowsHookEx(WH_KEYBOARD_LL, KeyboardHookProcedure, Marshal.GetHINSTANCE(Assembly.GetExecutingAssembly().GetModules()[0]), 0);
                //************************************
                //键盘线程钩子
                //SetWindowsHookEx( 2,KeyboardHookProcedure, IntPtr.Zero, GetCurrentThreadId());//指定要监听的线程idGetCurrentThreadId(),
                //键盘全局钩子,需要引用空间(using System.Reflection;)
                //SetWindowsHookEx( 13,MouseHookProcedure,Marshal.GetHINSTANCE(Assembly.GetExecutingAssembly().GetModules()[0]),0);
                //
                //关于SetWindowsHookEx (int idHook, HookProc lpfn, IntPtr hInstance, int threadId)函数将钩子加入到钩子链表中，说明一下四个参数：
                //idHook 钩子类型，即确定钩子监听何种消息，上面的代码中设为2，即监听键盘消息并且是线程钩子，如果是全局钩子监听键盘消息应设为13，
                //线程钩子监听鼠标消息设为7，全局钩子监听鼠标消息设为14。lpfn 钩子子程的地址指针。如果dwThreadId参数为0 或是一个由别的进程创建的
                //线程的标识，lpfn必须指向DLL中的钩子子程。 除此以外，lpfn可以指向当前进程的一段钩子子程代码。钩子函数的入口地址，当钩子钩到任何
                //消息后便调用这个函数。hInstance应用程序实例的句柄。标识包含lpfn所指的子程的DLL。如果threadId 标识当前进程创建的一个线程，而且子
                //程代码位于当前进程，hInstance必须为NULL。可以很简单的设定其为本应用程序的实例句柄。threaded 与安装的钩子子程相关联的线程的标识符
                //如果为0，钩子子程与所有的线程关联，即为全局钩子
                //************************************
                //如果SetWindowsHookEx失败
                if (hKeyboardHook == 0)
                {
                    Stop();
                    throw new Exception("安装键盘钩子失败");
                }
            }
        }
        public void Stop()
        {
            bool retKeyboard = true;
            if (hKeyboardHook != 0)
            {
                retKeyboard = UnhookWindowsHookEx(hKeyboardHook);
                hKeyboardHook = 0;
            }
            if (!(retKeyboard)) throw new Exception("卸载钩子失败！");
        }
        //ToAscii职能的转换指定的虚拟键码和键盘状态的相应字符或字符
        [DllImport("user32")]
        public static extern int ToAscii(int uVirtKey, //[in] 指定虚拟关键代码进行翻译。
                                         int uScanCode, // [in] 指定的硬件扫描码的关键须翻译成英文。高阶位的这个值设定的关键，如果是（不压）
                                         byte[] lpbKeyState, // [in] 指针，以256字节数组，包含当前键盘的状态。每个元素（字节）的数组包含状态的一个关键。如果高阶位的字节是一套，关键是下跌（按下）。在低比特，如果设置表明，关键是对切换。在此功能，只有肘位的CAPS LOCK键是相关的。在切换状态的NUM个锁和滚动锁定键被忽略。
                                         byte[] lpwTransKey, // [out] 指针的缓冲区收到翻译字符或字符。
                                         int fuState); // [in] Specifies whether a menu is active. This parameter must be 1 if a menu is active, or 0 otherwise.
        //获取按键的状态
        [DllImport("user32")]
        public static extern int GetKeyboardState(byte[] pbKeyState);
        [DllImport("user32.dll", CharSet = CharSet.Auto, CallingConvention = CallingConvention.StdCall)]
        private static extern short GetKeyState(int vKey);
        private const int WM_KEYDOWN = 0x100;//KEYDOWN
        private const int WM_KEYUP = 0x101;//KEYUP
        private const int WM_SYSKEYDOWN = 0x104;//SYSKEYDOWN
        private const int WM_SYSKEYUP = 0x105;//SYSKEYUP
        private int KeyboardHookProc(int nCode, Int32 wParam, IntPtr lParam)
        {
            // 侦听键盘事件
            if ((nCode >= 0) && (KeyDownEvent != null || KeyUpEvent != null || KeyPressEvent != null))
            {
                KeyboardHookStruct MyKeyboardHookStruct = (KeyboardHookStruct)Marshal.PtrToStructure(lParam, typeof(KeyboardHookStruct));
                // raise KeyDown
                if (KeyDownEvent != null && (wParam == WM_KEYDOWN || wParam == WM_SYSKEYDOWN))
                {
                    Keys keyData = (Keys)MyKeyboardHookStruct.vkCode;
                    System.Windows.Forms.KeyEventArgs e = new System.Windows.Forms.KeyEventArgs(keyData);
                    KeyDownEvent(this, e);
                }
                //键盘按下
                if (KeyPressEvent != null && wParam == WM_KEYDOWN)
                {
                    byte[] keyState = new byte[256];
                    GetKeyboardState(keyState);
                    byte[] inBuffer = new byte[2];
                    if (ToAscii(MyKeyboardHookStruct.vkCode, MyKeyboardHookStruct.scanCode, keyState, inBuffer, MyKeyboardHookStruct.flags) == 1)
                    {
                        KeyPressEventArgs e = new KeyPressEventArgs((char)inBuffer[0]);
                        KeyPressEvent(this, e);
                    }
                }
                // 键盘抬起
                if (KeyUpEvent != null && (wParam == WM_KEYUP || wParam == WM_SYSKEYUP))
                {
                    Keys keyData = (Keys)MyKeyboardHookStruct.vkCode;
                    System.Windows.Forms.KeyEventArgs e = new System.Windows.Forms.KeyEventArgs(keyData);
                    KeyUpEvent(this, e);
                }
            }
            //如果返回1，则结束消息，这个消息到此为止，不再传递。
            //如果返回0或调用CallNextHookEx函数则消息出了这个钩子继续往下传递，也就是传给消息真正的接受者
            return CallNextHookEx(hKeyboardHook, nCode, wParam, lParam);
        }
        ~KeyboardHook()
        {
            Stop();
        }
    }
    #endregion
}
