using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;
using System.Windows.Forms;

namespace WpfApplication1
{
    public partial class MainWindow : Window
    {
        
        public class Save2DSrcImageImpl : ICommand
        {
            public bool CanExecute(object parameter) { return true; }
            public event EventHandler CanExecuteChanged;

            public void Execute(object parameter)
            {
                if (c1.empty()) return;
                SaveFileDialog ofd = new SaveFileDialog();
                ofd.FileName = "";
                ofd.DefaultExt = "tif";
                ofd.Title = "Save2DSrcImage";
                if (ofd.ShowDialog().ToString().Equals("OK"))
                {
                    c1.saveSrc2DImage(ofd.FileName);
                }
            }
        }

        public class Save2DDstImageImpl : ICommand
        {
            public bool CanExecute(object parameter) { return true; }
            public event EventHandler CanExecuteChanged;

            public void Execute(object parameter)
            {
                if (c1.empty()) return;
                SaveFileDialog ofd = new SaveFileDialog();
                ofd.FileName = "";
                ofd.DefaultExt = "tif";
                ofd.Title = "Save2DDstImage";
                if (ofd.ShowDialog().ToString().Equals("OK"))
                {
                    c1.saveDst2DImage(ofd.FileName);
                }
            }
        }

        public class SaveHeightMapImpl : ICommand
        {
            public bool CanExecute(object parameter) { return true; }
            public event EventHandler CanExecuteChanged;

            public void Execute(object parameter)
            {
                if (c1.empty()) return;
                SaveFileDialog ofd = new SaveFileDialog();
                ofd.FileName = "";
                ofd.DefaultExt = "hmp";
                ofd.Title = "SaveHeightMap";
                if (ofd.ShowDialog().ToString().Equals("OK"))
                {
                    c1.saveHeightMap(ofd.FileName);
                }
            }
        }

        public class ReadHeightMapImpl : ICommand
        {
            public bool CanExecute(object parameter) { return true; }
            public event EventHandler CanExecuteChanged;

            public void Execute(object parameter)
            {
                if (c1.empty()) return;
                OpenFileDialog ofd = new OpenFileDialog();
                ofd.FileName = "";
                ofd.DefaultExt = "tif";
                ofd.Title = "ReadHeightMap";
                if (ofd.ShowDialog().ToString().Equals("OK"))
                {
                    c1.readHeightMap(ofd.FileName);
                }
            }
        }

        public class SelectionSliceCopyImpl : ICommand
        {
            public bool CanExecute(object parameter) { return true; }
            public event EventHandler CanExecuteChanged;

            public void Execute(object parameter)
            {
                //c1.setCopySelectionSliceSrcZX(-1);
            }
        }

        public class SelectionSlicePasteImpl : ICommand
        {
            public bool CanExecute(object parameter) { return true; }
            public event EventHandler CanExecuteChanged;

            public void Execute(object parameter)
            {
                //c1.copySelectionSliceZX(-1, -1);
            }
        }

        public class MaskImpl : ICommand
        {
            public bool CanExecute(object parameter) { return true; }
            public event EventHandler CanExecuteChanged;

            public void Execute(object parameter)
            {
                //c1.maskDstCall(false);
            }
        }

        public class MaskInvImpl : ICommand
        {
            public bool CanExecute(object parameter) { return true; }
            public event EventHandler CanExecuteChanged;

            public void Execute(object parameter)
            {
                //c1.maskDstCall(true);
            }
        }

        public class CalcAreaImpl : ICommand
        {
            public bool CanExecute(object parameter) { return true; }
            public event EventHandler CanExecuteChanged;

            public void Execute(object parameter)
            {
                c1.calcHmapAreaCall(10);
            }
        }

        public ICommand Save2DSrcImage { get; private set; }
        public ICommand Save2DDstImage { get; private set; }
        public ICommand SaveHeightMap { get; private set; }
        public ICommand ReadHeightMap { get; private set; }
        public ICommand SelectionSliceCopy { get; private set; }
        public ICommand SelectionSlicePaste { get; private set; }
        public ICommand Mask { get; private set; }
        public ICommand MaskInv { get; private set; }
        public ICommand CalcArea { get; private set; }
        public void InitCustomCommands()
        {
            Save2DSrcImage = new Save2DSrcImageImpl();
            Save2DDstImage = new Save2DDstImageImpl();
            SaveHeightMap = new SaveHeightMapImpl();
            ReadHeightMap = new ReadHeightMapImpl();
            SelectionSliceCopy = new SelectionSliceCopyImpl();
            SelectionSlicePaste = new SelectionSlicePasteImpl();
            Mask = new MaskImpl();
            MaskInv = new MaskInvImpl();
            CalcArea = new CalcAreaImpl();
            
        }
    }
}