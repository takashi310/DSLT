using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;

namespace WpfApplication1
{
    /// <summary>
    /// SubWindow.xaml の相互作用ロジック
    /// </summary>
    public partial class SubWindow : Window
    {
        public SubWindow()
        {
            InitializeComponent();
            this.Closing += new System.ComponentModel.CancelEventHandler(SubWindow_Closing);
            //System.Console.WriteLine("SUBWINDOW");
            
        }

        void SubWindow_Closing(object sender, CancelEventArgs e)
        {
            this.Hide();
            e.Cancel = true;
        }
    }
}
