
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
using System.Windows.Navigation;
using System.Windows.Shapes;
//using System.Windows.Forms;
using System.Diagnostics;
using TestCVclass;

namespace WpfApplication1
{
    /// <summary>
    /// MainWindow.xaml の相互作用ロジック
    /// </summary>
    public partial class MainWindow : Window
    {
        public static Class1 c1 = new Class1();
        private int imageW = -1;
        private int imageH = -1;
        private int imageZ = -1;
        private SubWindow sbw = new SubWindow();
        private Window1 win1 = new Window1();

        private double view_ratioW = -1;
        private double view_ratioH = -1;
        private double view_ratioD = -1;

        private double sv_im_ratioW = -1;
        private double sv_im_ratioH = -1;
        private double sv_im_ratioD = -1;

        private int im_mag_lv = 0;

        ListBoxItem dragItem;
        Point dragStartPos;
        DragAdorner dragGhost;

        public const int VIEW_ID_SRC = 0;
        public const int VIEW_ID_DST = 1;

        public MainWindow()
        {
            //sbw = new SubWindow();

            InitializeComponent();

            InitCustomCommands();
            AddHotKeys();

            this.DataContext = this;


        }

        private void AddHotKeys()
        {
            try
            {
                RoutedCommand deselectSegmentsCommand = new RoutedCommand();
                deselectSegmentsCommand.InputGestures.Add(new KeyGesture(Key.D, ModifierKeys.Alt));
                CommandBindings.Add(new CommandBinding(deselectSegmentsCommand, DeselectSegments_event_handler));

                RoutedCommand joinSegmentsCommand = new RoutedCommand();
                joinSegmentsCommand.InputGestures.Add(new KeyGesture(Key.J, ModifierKeys.Alt));
                CommandBindings.Add(new CommandBinding(joinSegmentsCommand, JoinSegments_event_handler));
                
                RoutedCommand separateSegmentsCommand = new RoutedCommand();
                separateSegmentsCommand.InputGestures.Add(new KeyGesture(Key.S, ModifierKeys.Alt));
                CommandBindings.Add(new CommandBinding(separateSegmentsCommand, SeparateSegments_event_handler));

                RoutedCommand loadBuckupSegDataCommand = new RoutedCommand();
                loadBuckupSegDataCommand.InputGestures.Add(new KeyGesture(Key.Z, ModifierKeys.Control));
                CommandBindings.Add(new CommandBinding(loadBuckupSegDataCommand, LoadBuckupSegData_event_handler));

                RoutedCommand selectAllSegCommand = new RoutedCommand();
                selectAllSegCommand.InputGestures.Add(new KeyGesture(Key.A, ModifierKeys.Alt));
                CommandBindings.Add(new CommandBinding(selectAllSegCommand, SelectAllSeg_event_handler));
            }
            catch (Exception err)
            {
                //handle exception error
                this.Close();
            }
        }

        private void DeselectSegments_event_handler(object sender, ExecutedRoutedEventArgs e)
        {
            c1.deselectSegment();
            updateImg();
        }

        private void JoinSegments_event_handler(object sender, ExecutedRoutedEventArgs e)
        {
            c1.joinSelectedSegments();
            updateImg();
        }

        private void SeparateSegments_event_handler(object sender, ExecutedRoutedEventArgs e)
        {
            c1.separateSelectedSegments();
            updateImg();
        }

        private void LoadBuckupSegData_event_handler(object sender, ExecutedRoutedEventArgs e)
        {
            c1.loadBackupSegmentsData();
            updateImg();
        }

        private void SelectAllSeg_event_handler(object sender, ExecutedRoutedEventArgs e)
        {
            c1.selectAllSegments();
            updateImg();
        }

        private void listBoxItem_PreviewMouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            // マウスダウンされたアイテムを記憶
            dragItem = sender as ListBoxItem;
            // マウスダウン時の座標を取得
            dragStartPos = e.GetPosition(dragItem);//ListBoxItem中の座標(ScaleTransformの影響を受けない)
        }

        //4～8行目:左マウスボタンが押されていて、ゴーストが null、sender が前項のアイテムと同じ、かつ、 マウス移動量がシステム設定値以上になったら、ドラッグのフェーズに入る。
        //10行目：リストボックスへのドロップを許可する。
        //12行目：まず、AdornerLayer.GetAdornerLayer メソッド（スタティック）でリストボックスの装飾レイヤーを得る。
        //13行目：オーナー、装飾オブジェクト、透明度、ドラッグ開始位置を渡して、ゴーストを初期化する。
        //14行目：ゴーストを装飾レイヤーへ追加
        //15行目：ドラッグドロップ処理を開始（ここで、ドロップされるまでブロックされる）
        //16行目：ゴーストを装飾レイヤーから削除
        //17～18行目：ゴーストとアイテムを初期化（null に）
        //20行目：リストボックスへのドロップを禁止する。
        private void listBoxItem_PreviewMouseMove(object sender, MouseEventArgs e)
        {
			var lbi = sender as ListBoxItem;
			
            if (e.LeftButton == MouseButtonState.Pressed && dragGhost == null && dragItem == lbi && lbi.IsKeyboardFocused)
            {
                var nowPos = e.GetPosition(lbi);
                if (Math.Abs(nowPos.X - dragStartPos.X) > SystemParameters.MinimumHorizontalDragDistance ||
                    Math.Abs(nowPos.Y - dragStartPos.Y) > SystemParameters.MinimumVerticalDragDistance)
                {
                    listBox.AllowDrop = true;
                    
                    var layer = AdornerLayer.GetAdornerLayer(listBox);
                    dragGhost = new DragAdorner(listBox, lbi, 0.5, dragStartPos);
                    layer.Add(dragGhost);
                    DragDrop.DoDragDrop(lbi, lbi, DragDropEffects.Move);
                    layer.Remove(dragGhost);
                    dragGhost = null;
                    dragItem = null;

                    listBox.AllowDrop = false;
                }
            }
        }

        private void listBoxItem_QueryContinueDrag(object sender, QueryContinueDragEventArgs e)
        {
            if (dragGhost != null)
            {
                var p = listBox.PointFromScreen(CursorInfo.GetNowPosition(this));
                dragGhost.LeftOffset = p.X;
                dragGhost.TopOffset = p.Y;
            }
        }
        
        private void listBox_Drop(object sender, DragEventArgs e)
        {
            var dropPos = e.GetPosition(listBox);
            var lbi = e.Data.GetData(typeof(ListBoxItem)) as ListBoxItem;
            var index = listBox.Items.IndexOf(lbi);
            for (int i = 0; i < listBox.Items.Count; i++)
            {
                var item = listBox.ItemContainerGenerator.ContainerFromIndex(i) as ListBoxItem;
                var pos = listBox.PointFromScreen(item.PointToScreen(new Point(0, item.ActualHeight / 2)));
                if (dropPos.Y < pos.Y)
                {
                    // i が入れ換え先のインデックス
                    listBox.Items.Remove(lbi);
                    listBox.Items.Insert((index < i) ? i - 1 : i, lbi);
                    debug_cmp_txt.Text = lbi.Name;
                    debug_txt.Text = ((index < i) ? i - 1 : i).ToString();
                    return;
                }
            }
            // 最後にもっていく
            int last = listBox.Items.Count - 1;
            listBox.Items.Remove(lbi);
            listBox.Items.Add(lbi);
            debug_cmp_txt.Text = lbi.Name;
            debug_txt.Text = last.ToString();
        }
       
        private void updateGraph()
        {
            if (c1.empty()) return;

            IntPtr data;
            int byteperpixel, width, height;

            if (c1.getXPlotGraph(out data, out byteperpixel, out width, out height) == 0) return;
            win1.graphX.Source = BitmapSource.Create(width, height, 96, 96, PixelFormats.Rgb24, null, data, width * height * byteperpixel, width * byteperpixel);
            if (c1.getYPlotGraph(out data, out byteperpixel, out width, out height) == 0) return;
            win1.graphY.Source = BitmapSource.Create(width, height, 96, 96, PixelFormats.Rgb24, null, data, width * height * byteperpixel, width * byteperpixel);
            if (c1.getZPlotGraph(out data, out byteperpixel, out width, out height) == 0) return;
            win1.graphZ.Source = BitmapSource.Create(width, height, 96, 96, PixelFormats.Rgb24, null, data, width * height * byteperpixel, width * byteperpixel);
        }

        private void updateImgXY()
        {
            if (c1.empty()) return;
            c1.setHmapActivity(H_hproj_cb.IsChecked.Value);
            if (H_normal_rb.IsChecked.Value) c1.setProjectionMode(0);
            else c1.setProjectionMode(1);
            c1.setDepthCodeEnable(DC_cb.IsChecked.Value);
			c1.setZBCEnable(ZBC_cb.IsChecked.Value);
            IntPtr data;
            int byteperpixel, width, height;
            //if (H_hproj_cb.IsChecked.Value) ;
            if (c1.getImageDataArrayXY(out data, out byteperpixel, out width, out height) == 0) return;
            ortho_xy.Source = BitmapSource.Create(width, height, 96, 96, PixelFormats.Rgb24, null, data, width * height * byteperpixel, width * byteperpixel);
            if (c1.getDstImageDataArrayXY(out data, out byteperpixel, out width, out height) == 0) return;
            dst_ortho_xy.Source = BitmapSource.Create(width, height, 96, 96, PixelFormats.Rgb24, null, data, width * height * byteperpixel, width * byteperpixel);
            
            updateGraph();
        }

        private void updateImgYZ()
        {
            if (c1.empty()) return;
            c1.setDepthCodeEnable(DC_cb.IsChecked.Value);
			c1.setZBCEnable(ZBC_cb.IsChecked.Value);
            IntPtr data;
            int byteperpixel, width, height;
            if (c1.getImageDataArrayYZ(out data, out byteperpixel, out width, out height) == 0) return;
            ortho_yz.Source = BitmapSource.Create(width, height, 96, 96, PixelFormats.Rgb24, null, data, width * height * byteperpixel, width * byteperpixel);
            if (c1.getDstImageDataArrayYZ(out data, out byteperpixel, out width, out height) == 0) return;
            dst_ortho_yz.Source = BitmapSource.Create(width, height, 96, 96, PixelFormats.Rgb24, null, data, width * height * byteperpixel, width * byteperpixel);
            
            updateGraph();
        }

        private void updateImgZX()
        {
            if (c1.empty()) return;
            c1.setDepthCodeEnable(DC_cb.IsChecked.Value);
			c1.setZBCEnable(ZBC_cb.IsChecked.Value);
            IntPtr data;
            int byteperpixel, width, height;
            if (c1.getImageDataArrayZX(out data, out byteperpixel, out width, out height) == 0) return;
            ortho_zx.Source = BitmapSource.Create(width, height, 96, 96, PixelFormats.Rgb24, null, data, width * height * byteperpixel, width * byteperpixel);
            if (c1.getDstImageDataArrayZX(out data, out byteperpixel, out width, out height) == 0) return;
            dst_ortho_zx.Source = BitmapSource.Create(width, height, 96, 96, PixelFormats.Rgb24, null, data, width * height * byteperpixel, width * byteperpixel);
            
            updateGraph();
        }

        public void updateImg()
        {
            updateImgXY();
            updateImgYZ();
            updateImgZX();
        }

        private void updateADTH()
        {
            
        }

        private void updateADTH2D()
        {
            
        }

        private void updateTH()
        {
            if (c1.empty()) return;
            updateImg();
        }

        private void updateTH2D()
        {
            if (c1.empty()) return;
            updateImg();
        }

        private void Iteration_slider_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {

        }

        private void inversion_Checked(object sender, RoutedEventArgs e)
        {
            debug_cmp_txt.Text = "INVERSION";
            debug_txt.Text = "Checked";
        }

        private void inversion_Unchecked(object sender, RoutedEventArgs e)
        {
            debug_cmp_txt.Text = "INVERSION";
            debug_txt.Text = "Unchecked";
        }

        private void X_slider_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            debug_cmp_txt.Text = "X_SLIDER";
            debug_txt.Text = e.NewValue.ToString("F4");
            c1.setX((int)(e.NewValue));
            updateImgYZ();
            if (line_xy_x != null && line_zx_x != null && dst_line_xy_x != null && dst_line_zx_x != null)
            {
                line_xy_x.X1 = (double)X_slider.Value / (double)X_slider.Maximum * (double)ortho_xy.ActualWidth;
                line_xy_x.X2 = (double)X_slider.Value / (double)X_slider.Maximum * (double)ortho_xy.ActualWidth;
                line_xy_x.Y1 = 0;
                line_xy_x.Y2 = ortho_xy.ActualHeight;

                line_zx_x.X1 = (double)X_slider.Value / (double)X_slider.Maximum * (double)ortho_zx.ActualWidth;
                line_zx_x.X2 = (double)X_slider.Value / (double)X_slider.Maximum * (double)ortho_zx.ActualWidth;
                line_zx_x.Y1 = 0;
                line_zx_x.Y2 = ortho_zx.ActualHeight;

                dst_line_xy_x.X1 = (double)X_slider.Value / (double)X_slider.Maximum * (double)dst_ortho_xy.ActualWidth;
                dst_line_xy_x.X2 = (double)X_slider.Value / (double)X_slider.Maximum * (double)dst_ortho_xy.ActualWidth;
                dst_line_xy_x.Y1 = 0;
                dst_line_xy_x.Y2 = dst_ortho_xy.ActualHeight;

                dst_line_zx_x.X1 = (double)X_slider.Value / (double)X_slider.Maximum * (double)dst_ortho_zx.ActualWidth;
                dst_line_zx_x.X2 = (double)X_slider.Value / (double)X_slider.Maximum * (double)dst_ortho_zx.ActualWidth;
                dst_line_zx_x.Y1 = 0;
                dst_line_zx_x.Y2 = dst_ortho_zx.ActualHeight;
            }
        }

        private void Y_slider_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            debug_cmp_txt.Text = "Y_SLIDER";
            debug_txt.Text = e.NewValue.ToString("F4");
            c1.setY((int)(e.NewValue));
            updateImgZX();
            if (line_xy_y != null && line_yz_y != null && dst_line_xy_y != null && dst_line_yz_y != null)
            {
                line_xy_y.Y1 = (double)Y_slider.Value / (double)Y_slider.Maximum * (double)ortho_xy.ActualHeight;
                line_xy_y.Y2 = (double)Y_slider.Value / (double)Y_slider.Maximum * (double)ortho_xy.ActualHeight;
                line_xy_y.X1 = 0;
                line_xy_y.X2 = ortho_xy.ActualWidth;

                line_yz_y.Y1 = (double)Y_slider.Value / (double)Y_slider.Maximum * (double)ortho_yz.ActualHeight;
                line_yz_y.Y2 = (double)Y_slider.Value / (double)Y_slider.Maximum * (double)ortho_yz.ActualHeight;
                line_yz_y.X1 = 0;
                line_yz_y.X2 = ortho_yz.ActualWidth;

                dst_line_xy_y.Y1 = (double)Y_slider.Value / (double)Y_slider.Maximum * (double)dst_ortho_xy.ActualHeight;
                dst_line_xy_y.Y2 = (double)Y_slider.Value / (double)Y_slider.Maximum * (double)dst_ortho_xy.ActualHeight;
                dst_line_xy_y.X1 = 0;
                dst_line_xy_y.X2 = dst_ortho_xy.ActualWidth;

                dst_line_yz_y.Y1 = (double)Y_slider.Value / (double)Y_slider.Maximum * (double)dst_ortho_yz.ActualHeight;
                dst_line_yz_y.Y2 = (double)Y_slider.Value / (double)Y_slider.Maximum * (double)dst_ortho_yz.ActualHeight;
                dst_line_yz_y.X1 = 0;
                dst_line_yz_y.X2 = dst_ortho_yz.ActualWidth;
            }
        }

        private void Z_slider_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            debug_cmp_txt.Text = "Z_SLIDER";
            debug_txt.Text = e.NewValue.ToString("F4");
            c1.setZ((int)(e.NewValue));
            updateImgXY();
            if (line_yz_z != null && line_zx_z != null && dst_line_yz_z != null && dst_line_zx_z != null)
            {
                line_yz_z.X1 = (double)Z_slider.Value / (double)Z_slider.Maximum * (double)ortho_yz.ActualWidth;
                line_yz_z.X2 = (double)Z_slider.Value / (double)Z_slider.Maximum * (double)ortho_yz.ActualWidth;
                line_yz_z.Y1 = 0;
                line_yz_z.Y2 = ortho_yz.ActualHeight;

                line_zx_z.X1 = 0;
                line_zx_z.X2 = ortho_zx.ActualWidth;
                line_zx_z.Y1 = (double)Z_slider.Value / (double)Z_slider.Maximum * (double)ortho_zx.ActualHeight;
                line_zx_z.Y2 = (double)Z_slider.Value / (double)Z_slider.Maximum * (double)ortho_zx.ActualHeight;

                dst_line_yz_z.X1 = (double)Z_slider.Value / (double)Z_slider.Maximum * (double)dst_ortho_yz.ActualWidth;
                dst_line_yz_z.X2 = (double)Z_slider.Value / (double)Z_slider.Maximum * (double)dst_ortho_yz.ActualWidth;
                dst_line_yz_z.Y1 = 0;
                dst_line_yz_z.Y2 = dst_ortho_yz.ActualHeight;

                dst_line_zx_z.X1 = 0;
                dst_line_zx_z.X2 = dst_ortho_zx.ActualWidth;
                dst_line_zx_z.Y1 = (double)Z_slider.Value / (double)Z_slider.Maximum * (double)dst_ortho_zx.ActualHeight;
                dst_line_zx_z.Y2 = (double)Z_slider.Value / (double)Z_slider.Maximum * (double)dst_ortho_zx.ActualHeight;
            }
        }

		private void init()
		{
			if (c1.empty()) return;

            //imageZ == 1だと0除算エラー(maximum = 1, slider初期値 = 0に設定して非アクティブにすればよいかも)
			updateImgXY();
			updateImgYZ();
			updateImgZX();
			c1.getImageSize(out imageW, out imageH, out imageZ);
			
            X_slider.Value = 0;
            X_slider.Maximum = imageW > 1 ? imageW - 1 : 1;
            X_slider.IsEnabled = imageW > 1 ? true : false;
            X_txt.IsEnabled = imageW > 1 ? true : false;
			
            Y_slider.Value = 0;
            Y_slider.Maximum = imageH > 1 ? imageH - 1 : 1;
            Y_slider.IsEnabled = imageH > 1 ? true : false;
            Y_txt.IsEnabled = imageH > 1 ? true : false;
			
            Z_slider.Value = 0;
            Z_slider.Maximum = imageZ > 1 ? imageZ - 1 : 1;
            Z_slider.IsEnabled = imageZ > 1 ? true : false;
            Z_txt.IsEnabled = imageZ > 1 ? true : false;

            c1.setX(0);
            c1.setY(0);
            c1.setZ(0);

            H_offset_slider.Value = 0;
            H_offset_slider.Maximum = imageZ > 1 ? imageZ - 1 : 1;
			H_offset_slider.Minimum = -(imageZ > 1 ? imageZ - 1 : 1);
            H_offset_slider.IsEnabled = imageZ > 1 ? true : false;
            H_offset_txt.IsEnabled = imageZ > 1 ? true : false;

            H_depth_slider.Value = 0;
            H_depth_slider.Maximum = imageZ > 1 ? imageZ - 1 : 1;
            H_depth_slider.Minimum = -(imageZ > 1 ? imageZ - 1 : 1);
            H_depth_slider.IsEnabled = imageZ > 1 ? true : false;
            H_depth_txt.IsEnabled = imageZ > 1 ? true : false;

            H_range_slider.Value = 0;
            H_range_slider.Maximum = imageZ > 1 ? imageZ - 1 : 1;
            H_range_slider.IsEnabled = imageZ > 1 ? true : false;
            H_range_txt.IsEnabled = imageZ > 1 ? true : false;

            H_upper_slider.Value = 0;
            H_upper_slider.Maximum = imageZ > 1 ? imageZ - 1 : 1;
            H_upper_slider.Minimum = -(imageZ > 1 ? imageZ - 1 : 1);
            H_upper_slider.IsEnabled = imageZ > 1 ? true : false;
            H_upper_txt.IsEnabled = imageZ > 1 ? true : false;

            H_lower_slider.Value = 0;
            H_lower_slider.Maximum = imageZ > 1 ? imageZ - 1 : 1;
            H_lower_slider.Minimum = -(imageZ > 1 ? imageZ - 1 : 1);
            H_lower_slider.IsEnabled = imageZ > 1 ? true : false;
            H_lower_txt.IsEnabled = imageZ > 1 ? true : false;

            H_halo_slider.Value = 0;
            int xy_max = imageW > imageH ? imageW : imageH;
            H_halo_slider.Maximum = xy_max > 1 ? xy_max - 1 : 1;
            H_halo_slider.Minimum = 0;
            H_halo_slider.IsEnabled = xy_max > 1 ? true : false;
            H_halo_txt.IsEnabled = xy_max > 1 ? true : false;

			ch_slider.Maximum = c1.getChannelNum() > 1 ? c1.getChannelNum() - 1 : 1;
            ch_slider.Value = 0;
            ch_slider.IsEnabled = c1.getChannelNum() > 1 ? true : false;
            ch_txt.IsEnabled = c1.getChannelNum() > 1 ? true : false;
           
            ZBC_ch_slider.Maximum = c1.getChannelNum() > 1 ? c1.getChannelNum() - 1 : 1;
            ZBC_ch_slider.Value = 0;
            ZBC_ch_slider.IsEnabled = c1.getChannelNum() > 1 ? true : false;
            ZBC_ch_txt.IsEnabled = c1.getChannelNum() > 1 ? true : false;

			c1.setBCmax((float)BC_max_slider.Value);
			c1.setBCmin((float)BC_min_slider.Value);
			c1.setProjectionThreshold((float)H_proj_th_slider.Value);
			c1.setHmapOffset((float)H_offset_slider.Value);
			c1.setHmapProjDepth((float)H_depth_slider.Value);
			c1.setHmapProjRange((float)H_range_slider.Value);
			c1.setDmapRange((float)DC_range_slider.Value);
			c1.setHmapActivity(H_hproj_cb.IsChecked.Value);
            c1.setHmapVisibility(H_show_cb.IsChecked.Value);
			if (H_normal_rb.IsChecked.Value) c1.setProjectionMode(0);
			else c1.setProjectionMode(1);
			c1.setDepthCodeEnable(DC_cb.IsChecked.Value);
			c1.setDmapCoefficient((float)DC_coefficient_slider.Value);
			c1.setDmapOrder((float)DC_order_slider.Value);
			c1.setZBCEnable(ZBC_cb.IsChecked.Value);
			c1.setZBCparams((int)ZBC_ch_slider.Value, (float)ZBC_coefficient_slider.Value, (float)ZBC_order_slider.Value);
			c1.setSegMinVol((int)SG_min_size_slider.Value);
			c1.setSrcSegVisibility((bool)SG_src_cb.IsChecked.Value);
            c1.setDstSegVisibility((bool)SG_dst_cb.IsChecked.Value);
            c1.setCroppingParams(H_crop_cb.IsChecked.Value, H_crop_hmap_cb.IsChecked.Value, (int)H_upper_slider.Value, (int)H_lower_slider.Value, (int)H_halo_slider.Value);


            view_ratioW = (double)imageW;
            view_ratioH = (double)imageH;
            view_ratioD = (double)imageZ;

            sv_im_ratioW = 1.0;
            sv_im_ratioH = 1.0;
            sv_im_ratioD = 1.0;

            im_mag_lv = 0;

			src_sv_xy.Visibility = Visibility.Visible;
			src_sv_yz.Visibility = Visibility.Visible;
			src_sv_zx.Visibility = Visibility.Visible;
			dst_sv_xy.Visibility = Visibility.Visible;
			dst_sv_yz.Visibility = Visibility.Visible;
			dst_sv_zx.Visibility = Visibility.Visible;

            ortho_grid_src.ColumnDefinitions.ElementAt(2).Width = new GridLength((double)imageZ / (double)imageW, GridUnitType.Star);
            ortho_grid_dst.ColumnDefinitions.ElementAt(2).Width = new GridLength((double)imageZ / (double)imageW, GridUnitType.Star);
            ortho_grid_src.RowDefinitions.ElementAt(2).Height = new GridLength((double)imageZ / (double)imageH, GridUnitType.Star);
            ortho_grid_dst.RowDefinitions.ElementAt(2).Height = new GridLength((double)imageZ / (double)imageH, GridUnitType.Star);

            line_xy_x.X1 = (double)X_slider.Value / (double)X_slider.Maximum * (double)ortho_xy.ActualWidth;
            line_xy_x.X2 = (double)X_slider.Value / (double)X_slider.Maximum * (double)ortho_xy.ActualWidth;
            line_xy_x.Y1 = 0;
            line_xy_x.Y2 = ortho_xy.ActualHeight;
            line_xy_x.StrokeThickness = 1;
            line_xy_x.Visibility = Visibility.Visible;

            line_xy_y.X1 = 0;
            line_xy_y.X2 = ortho_xy.ActualWidth;
            line_xy_y.Y1 = (double)Y_slider.Value / (double)Y_slider.Maximum * (double)ortho_xy.ActualHeight;
            line_xy_y.Y2 = (double)Y_slider.Value / (double)Y_slider.Maximum * (double)ortho_xy.ActualHeight;
            line_xy_y.StrokeThickness = 1;
            line_xy_y.Visibility = Visibility.Visible;

            line_yz_y.X1 = 0;
            line_yz_y.X2 = ortho_yz.ActualWidth;
            line_yz_y.Y1 = (double)Y_slider.Value / (double)Y_slider.Maximum * (double)ortho_yz.ActualHeight;
            line_yz_y.Y2 = (double)Y_slider.Value / (double)Y_slider.Maximum * (double)ortho_yz.ActualHeight;
            line_yz_y.StrokeThickness = 1;
            line_yz_y.Visibility = Visibility.Visible;

            line_yz_z.X1 = (double)Z_slider.Value / (double)Z_slider.Maximum * (double)ortho_yz.ActualWidth;
            line_yz_z.X2 = (double)Z_slider.Value / (double)Z_slider.Maximum * (double)ortho_yz.ActualWidth;
            line_yz_z.Y1 = 0;
            line_yz_z.Y2 = ortho_yz.ActualHeight;
            line_yz_z.StrokeThickness = 1;
            line_yz_z.Visibility = Visibility.Visible;

            line_zx_x.Y1 = 0;
            line_zx_x.Y2 = ortho_zx.ActualHeight;
            line_zx_x.X1 = (double)X_slider.Value / (double)X_slider.Maximum * (double)ortho_zx.ActualWidth;
            line_zx_x.X2 = (double)X_slider.Value / (double)X_slider.Maximum * (double)ortho_zx.ActualWidth;
            line_zx_x.StrokeThickness = 1;
            line_zx_x.Visibility = Visibility.Visible;

            line_zx_z.X1 = 0;
            line_zx_z.X2 = ortho_zx.ActualWidth;
            line_zx_z.Y1 = (double)Z_slider.Value / (double)Z_slider.Maximum * (double)ortho_zx.ActualHeight;
            line_zx_z.Y2 = (double)Z_slider.Value / (double)Z_slider.Maximum * (double)ortho_zx.ActualHeight;
            line_zx_z.StrokeThickness = 1;
            line_zx_z.Visibility = Visibility.Visible;

            dst_line_xy_x.X1 = (double)X_slider.Value / (double)X_slider.Maximum * (double)dst_ortho_xy.ActualWidth;
            dst_line_xy_x.X2 = (double)X_slider.Value / (double)X_slider.Maximum * (double)dst_ortho_xy.ActualWidth;
            dst_line_xy_x.Y1 = 0;
            dst_line_xy_x.Y2 = dst_ortho_xy.ActualHeight;
            dst_line_xy_x.StrokeThickness = 1;
            dst_line_xy_x.Visibility = Visibility.Visible;

            dst_line_xy_y.X1 = 0;
            dst_line_xy_y.X2 = dst_ortho_xy.ActualWidth;
            dst_line_xy_y.Y1 = (double)Y_slider.Value / (double)Y_slider.Maximum * (double)dst_ortho_xy.ActualHeight;
            dst_line_xy_y.Y2 = (double)Y_slider.Value / (double)Y_slider.Maximum * (double)dst_ortho_xy.ActualHeight;
            dst_line_xy_y.StrokeThickness = 1;
            dst_line_xy_y.Visibility = Visibility.Visible;

            dst_line_yz_y.X1 = 0;
            dst_line_yz_y.X2 = ortho_yz.ActualWidth;
            dst_line_yz_y.Y1 = (double)Y_slider.Value / (double)Y_slider.Maximum * (double)dst_ortho_yz.ActualHeight;
            dst_line_yz_y.Y2 = (double)Y_slider.Value / (double)Y_slider.Maximum * (double)dst_ortho_yz.ActualHeight;
            dst_line_yz_y.StrokeThickness = 1;
            dst_line_yz_y.Visibility = Visibility.Visible;

            dst_line_yz_z.X1 = (double)Z_slider.Value / (double)Z_slider.Maximum * (double)dst_ortho_yz.ActualWidth;
            dst_line_yz_z.X2 = (double)Z_slider.Value / (double)Z_slider.Maximum * (double)dst_ortho_yz.ActualWidth;
            dst_line_yz_z.Y1 = 0;
            dst_line_yz_z.Y2 = ortho_yz.ActualHeight;
            dst_line_yz_z.StrokeThickness = 1;
            dst_line_yz_z.Visibility = Visibility.Visible;

            dst_line_zx_x.Y1 = 0;
            dst_line_zx_x.Y2 = ortho_zx.ActualHeight;
            dst_line_zx_x.X1 = (double)X_slider.Value / (double)X_slider.Maximum * (double)dst_ortho_zx.ActualWidth;
            dst_line_zx_x.X2 = (double)X_slider.Value / (double)X_slider.Maximum * (double)dst_ortho_zx.ActualWidth;
            dst_line_zx_x.StrokeThickness = 1;
            dst_line_zx_x.Visibility = Visibility.Visible;

            dst_line_zx_z.X1 = 0;
            dst_line_zx_z.X2 = ortho_zx.ActualWidth;
            dst_line_zx_z.Y1 = (double)Z_slider.Value / (double)Z_slider.Maximum * (double)dst_ortho_zx.ActualHeight;
            dst_line_zx_z.Y2 = (double)Z_slider.Value / (double)Z_slider.Maximum * (double)dst_ortho_zx.ActualHeight;
            dst_line_zx_z.StrokeThickness = 1;
            dst_line_zx_z.Visibility = Visibility.Visible;


            ortho_view.UpdateLayout();
            viewSizeChange(view_src);
            viewSizeChange(view_dst);

			win1.Hide();
			win1.graphX.Stretch = Stretch.None;
			win1.graphY.Stretch = Stretch.None;
			win1.graphZ.Stretch = Stretch.None;
			win1.Show();
			win1.SizeToContent = SizeToContent.WidthAndHeight;
			win1.SizeToContent = SizeToContent.Manual;
			win1.graphX.Stretch = Stretch.Uniform;
			win1.graphY.Stretch = Stretch.Uniform;
			win1.graphZ.Stretch = Stretch.Uniform;

        }

        private void Open_Executed(object sender, ExecutedRoutedEventArgs e)
        {
            var ofd = new System.Windows.Forms.OpenFileDialog();
            ofd.FileName = "";
            ofd.DefaultExt = "*.tif";
            if (ofd.ShowDialog().ToString().Equals("OK"))
            {
                debug_cmp_txt.Text = "MENU_OPEN";
                debug_txt.Text = ofd.FileName;
				
				ch_slider.Value = 0;
				c1.set3DImage_MultiTIFF(ofd.FileName, getCurrentChannel(), getCurrentScalingType());

				init();
            }
        }

        private void Save_Executed(object sender, ExecutedRoutedEventArgs e)
        {
            if (c1.empty()) return;

            var ofd = new System.Windows.Forms.SaveFileDialog();
            ofd.FileName = "";
            //ofd.DefaultExt = "tif";
            if (ofd.ShowDialog().ToString().Equals("OK"))
            {
                c1.saveDst3DImage(ofd.FileName);
            }
        }

        private void LoadSegments_Click(object sender, RoutedEventArgs e)
        {
            if (c1.empty()) return;

            var ofd = new System.Windows.Forms.OpenFileDialog();
            ofd.FileName = "";
            ofd.DefaultExt = "*.tif";
            ofd.Title = "Load Segments";
            if (ofd.ShowDialog().ToString().Equals("OK"))
            {
                c1.loadSegments(ofd.FileName);
                updateImg();
            }
        }

        private void SaveSegments_Click(object sender, RoutedEventArgs e)
        {
            if (c1.empty()) return;

            var ofd = new System.Windows.Forms.SaveFileDialog();
            ofd.FileName = "";
            //ofd.DefaultExt = "*.tif";
            ofd.Title = "Save Segments";
            if (ofd.ShowDialog().ToString().Equals("OK"))
            {
                c1.saveSegments(ofd.FileName);
                updateImg();
            }
        }

        private void Close_Executed(object sender, ExecutedRoutedEventArgs e)
        {
            this.Close();
        }

        private void ortho_grid_SizeChanged(object sender, SizeChangedEventArgs e)
        {
            
        }

        private void ortho_xy_SizeChanged(object sender, SizeChangedEventArgs e)
        {
            line_xy_x.X1 = (double)X_slider.Value / (double)X_slider.Maximum * (double)((Image)sender).ActualWidth;
            line_xy_x.X2 = (double)X_slider.Value / (double)X_slider.Maximum * (double)((Image)sender).ActualWidth;
            line_xy_x.Y1 = 0;
            line_xy_x.Y2 = ((Image)sender).ActualHeight;

            line_xy_y.Y1 = (double)Y_slider.Value / (double)Y_slider.Maximum * (double)((Image)sender).ActualHeight;
            line_xy_y.Y2 = (double)Y_slider.Value / (double)Y_slider.Maximum * (double)((Image)sender).ActualHeight;
            line_xy_y.X1 = 0;
            line_xy_y.X2 = ((Image)sender).ActualWidth;

        }

        private void ortho_yz_SizeChanged(object sender, SizeChangedEventArgs e)
        {
            line_yz_y.X1 = 0;
            line_yz_y.X2 = ((Image)sender).ActualWidth;
            line_yz_y.Y1 = (double)Y_slider.Value / (double)Y_slider.Maximum * (double)((Image)sender).ActualHeight;
            line_yz_y.Y2 = (double)Y_slider.Value / (double)Y_slider.Maximum * (double)((Image)sender).ActualHeight;

            line_yz_z.X1 = (double)Z_slider.Value / (double)Z_slider.Maximum * (double)((Image)sender).ActualWidth;
            line_yz_z.X2 = (double)Z_slider.Value / (double)Z_slider.Maximum * (double)((Image)sender).ActualWidth;
            line_yz_z.Y1 = 0;
            line_yz_z.Y2 = ((Image)sender).ActualHeight;
        }

        private void ortho_zx_SizeChanged(object sender, SizeChangedEventArgs e)
        {
            line_zx_z.X1 = 0;
            line_zx_z.X2 = ((Image)sender).ActualWidth;
            line_zx_z.Y1 = (double)Z_slider.Value / (double)Z_slider.Maximum * (double)((Image)sender).ActualHeight;
            line_zx_z.Y2 = (double)Z_slider.Value / (double)Z_slider.Maximum * (double)((Image)sender).ActualHeight;

            line_zx_x.Y1 = 0;
            line_zx_x.Y2 = ((Image)sender).ActualHeight;
            line_zx_x.X1 = (double)X_slider.Value / (double)X_slider.Maximum * (double)((Image)sender).ActualWidth;
            line_zx_x.X2 = (double)X_slider.Value / (double)X_slider.Maximum * (double)((Image)sender).ActualWidth;
        }

        private void dst_ortho_xy_SizeChanged(object sender, SizeChangedEventArgs e)
        {
            dst_line_xy_x.X1 = (double)X_slider.Value / (double)X_slider.Maximum * (double)((Image)sender).ActualWidth;
            dst_line_xy_x.X2 = (double)X_slider.Value / (double)X_slider.Maximum * (double)((Image)sender).ActualWidth;
            dst_line_xy_x.Y1 = 0;
            dst_line_xy_x.Y2 = ((Image)sender).ActualHeight;

            dst_line_xy_y.Y1 = (double)Y_slider.Value / (double)Y_slider.Maximum * (double)((Image)sender).ActualHeight;
            dst_line_xy_y.Y2 = (double)Y_slider.Value / (double)Y_slider.Maximum * (double)((Image)sender).ActualHeight;
            dst_line_xy_y.X1 = 0;
            dst_line_xy_y.X2 = ((Image)sender).ActualWidth;
        }

        private void dst_ortho_yz_SizeChanged(object sender, SizeChangedEventArgs e)
        {
            dst_line_yz_y.X1 = 0;
            dst_line_yz_y.X2 = ((Image)sender).ActualWidth;
            dst_line_yz_y.Y1 = (double)Y_slider.Value / (double)Y_slider.Maximum * (double)((Image)sender).ActualHeight;
            dst_line_yz_y.Y2 = (double)Y_slider.Value / (double)Y_slider.Maximum * (double)((Image)sender).ActualHeight;

            dst_line_yz_z.X1 = (double)Z_slider.Value / (double)Z_slider.Maximum * (double)((Image)sender).ActualWidth;
            dst_line_yz_z.X2 = (double)Z_slider.Value / (double)Z_slider.Maximum * (double)((Image)sender).ActualWidth;
            dst_line_yz_z.Y1 = 0;
            dst_line_yz_z.Y2 = ((Image)sender).ActualHeight;
        }

        private void dst_ortho_zx_SizeChanged(object sender, SizeChangedEventArgs e)
        {
            dst_line_zx_z.X1 = 0;
            dst_line_zx_z.X2 = ((Image)sender).ActualWidth;
            dst_line_zx_z.Y1 = (double)Z_slider.Value / (double)Z_slider.Maximum * (double)((Image)sender).ActualHeight;
            dst_line_zx_z.Y2 = (double)Z_slider.Value / (double)Z_slider.Maximum * (double)((Image)sender).ActualHeight;

            dst_line_zx_x.Y1 = 0;
            dst_line_zx_x.Y2 = ((Image)sender).ActualHeight;
            dst_line_zx_x.X1 = (double)X_slider.Value / (double)X_slider.Maximum * (double)((Image)sender).ActualWidth;
            dst_line_zx_x.X2 = (double)X_slider.Value / (double)X_slider.Maximum * (double)((Image)sender).ActualWidth;
        }

        private void segmentSelection(int x, int y, int z)
        {

            if (SG_src_cb.IsChecked.Equals(true) || SG_dst_cb.IsChecked.Equals(true))
            {
                c1.selectSegment(x, y, z);
                updateImg();
            }

        }

        private void segmentSelectionXY(int x, int y, int z)
        {
            if (SG_src_cb.IsChecked.Equals(true) || SG_dst_cb.IsChecked.Equals(true))
            {
                c1.selectSegment_OrthoXY(x, y, z);
                updateImg();
            }

        }

        private void line_xy_MouseAction(object sender, System.Windows.Input.MouseEventArgs e)
        {
            if (Keyboard.IsKeyDown(Key.LeftShift) || Keyboard.IsKeyDown(Key.RightShift)) return;
            if (e.LeftButton.Equals(MouseButtonState.Pressed))
            {
                X_slider.Value = (int)(e.GetPosition(ortho_xy).X / ortho_xy.ActualWidth * (double)imageW);
                Y_slider.Value = (int)(e.GetPosition(ortho_xy).Y / ortho_xy.ActualHeight * (double)imageH);
            }
            if (e.MiddleButton.Equals(MouseButtonState.Pressed))
            {
                int x = (int)(e.GetPosition(ortho_xy).X / ortho_xy.ActualWidth * (double)imageW);
                int y = (int)(e.GetPosition(ortho_xy).Y / ortho_xy.ActualHeight * (double)imageH);
                int z = (int)Z_slider.Value;
                segmentSelectionXY(x, y, z);
                //segmentSelection(x, y, z);
            }
        }

        private void line_zx_MouseAction(object sender, System.Windows.Input.MouseEventArgs e)
        {
            if (Keyboard.IsKeyDown(Key.LeftShift) || Keyboard.IsKeyDown(Key.RightShift)) return;
            if (e.LeftButton.Equals(MouseButtonState.Pressed))
            {
                X_slider.Value = (int)(e.GetPosition(ortho_zx).X / ortho_zx.ActualWidth * (double)imageW);
                Z_slider.Value = (int)(e.GetPosition(ortho_zx).Y / ortho_zx.ActualHeight * (double)imageZ);
            }
            if (e.MiddleButton.Equals(MouseButtonState.Pressed))
            {
                int x = (int)(e.GetPosition(ortho_zx).X / ortho_zx.ActualWidth * (double)imageW);
                int y = (int)Y_slider.Value;
                int z = (int)(e.GetPosition(ortho_zx).Y / ortho_zx.ActualHeight * (double)imageZ);
                segmentSelection(x, y, z);
            }
        }

        private void line_yz_MouseAction(object sender, System.Windows.Input.MouseEventArgs e)
        {
            if (Keyboard.IsKeyDown(Key.LeftShift) || Keyboard.IsKeyDown(Key.RightShift)) return;
            if (e.LeftButton.Equals(MouseButtonState.Pressed))
            {
                Z_slider.Value = (int)(e.GetPosition(ortho_yz).X / (ortho_yz).ActualWidth * (double)imageZ);
                Y_slider.Value = (int)(e.GetPosition(ortho_yz).Y / ortho_yz.ActualHeight * (double)imageH);
            }
            if (e.MiddleButton.Equals(MouseButtonState.Pressed))
            {
                int x = (int)X_slider.Value;
                int y = (int)(e.GetPosition(ortho_yz).Y / ortho_yz.ActualHeight * (double)imageH);
                int z = (int)(e.GetPosition(ortho_yz).X / (ortho_yz).ActualWidth * (double)imageZ);
                segmentSelection(x, y, z);
            }
        }

        private void dst_line_xy_MouseAction(object sender, System.Windows.Input.MouseEventArgs e)
        {
            if (Keyboard.IsKeyDown(Key.LeftShift) || Keyboard.IsKeyDown(Key.RightShift)) return;
            if (e.LeftButton.Equals(MouseButtonState.Pressed))
            {
                X_slider.Value = (int)(e.GetPosition(dst_ortho_xy).X / dst_ortho_xy.ActualWidth * (double)imageW);
                Y_slider.Value = (int)(e.GetPosition(dst_ortho_xy).Y / dst_ortho_xy.ActualHeight * (double)imageH);
            }
            if (e.MiddleButton.Equals(MouseButtonState.Pressed))
            {
                int x = (int)(e.GetPosition(dst_ortho_xy).X / dst_ortho_xy.ActualWidth * (double)imageW);
                int y = (int)(e.GetPosition(dst_ortho_xy).Y / dst_ortho_xy.ActualHeight * (double)imageH);
                int z = (int)Z_slider.Value;
                segmentSelectionXY(x, y, z);
                //segmentSelection(x, y, z);
            }
        }

        private void dst_line_zx_MouseAction(object sender, System.Windows.Input.MouseEventArgs e)
        {
            if (Keyboard.IsKeyDown(Key.LeftShift) || Keyboard.IsKeyDown(Key.RightShift)) return;
            if (e.LeftButton.Equals(MouseButtonState.Pressed))
            {
                X_slider.Value = (int)(e.GetPosition(dst_ortho_zx).X / dst_ortho_zx.ActualWidth * (double)imageW);
                Z_slider.Value = (int)(e.GetPosition(dst_ortho_zx).Y / dst_ortho_zx.ActualHeight * (double)imageZ);
            }
            if (e.MiddleButton.Equals(MouseButtonState.Pressed))
            {
                int x = (int)(e.GetPosition(dst_ortho_zx).X / dst_ortho_zx.ActualWidth * (double)imageW);
                int y = (int)Y_slider.Value;
                int z = (int)(e.GetPosition(dst_ortho_zx).Y / dst_ortho_zx.ActualHeight * (double)imageZ);
                segmentSelection(x, y, z);
            }
        }

        private void dst_line_yz_MouseAction(object sender, System.Windows.Input.MouseEventArgs e)
        {
            if (Keyboard.IsKeyDown(Key.LeftShift) || Keyboard.IsKeyDown(Key.RightShift)) return;
            if (e.LeftButton.Equals(MouseButtonState.Pressed))
            {
                Z_slider.Value = (int)(e.GetPosition(dst_ortho_yz).X / (dst_ortho_yz).ActualWidth * (double)imageZ);
                Y_slider.Value = (int)(e.GetPosition(dst_ortho_yz).Y / dst_ortho_yz.ActualHeight * (double)imageH);
            }
            if (e.MiddleButton.Equals(MouseButtonState.Pressed))
            {
                int x = (int)X_slider.Value;
                int y = (int)(e.GetPosition(dst_ortho_yz).Y / dst_ortho_yz.ActualHeight * (double)imageH);
                int z = (int)(e.GetPosition(dst_ortho_yz).X / (dst_ortho_yz).ActualWidth * (double)imageZ);
                segmentSelection(x, y, z);
            }
        }

        private void ortho_xy_MouseAction(object sender, System.Windows.Input.MouseEventArgs e)
        {
            if (Keyboard.IsKeyDown(Key.LeftShift) || Keyboard.IsKeyDown(Key.RightShift)) return;
            if (e.LeftButton.Equals(MouseButtonState.Pressed))
            {
                X_slider.Value = (int)(e.GetPosition(((Image)sender)).X / ((Image)sender).ActualWidth * (double)imageW);
                Y_slider.Value = (int)(e.GetPosition(((Image)sender)).Y / ((Image)sender).ActualHeight * (double)imageH);
            }
            if (e.MiddleButton.Equals(MouseButtonState.Pressed))
            {
                int x = (int)(e.GetPosition(((Image)sender)).X / ((Image)sender).ActualWidth * (double)imageW);
                int y = (int)(e.GetPosition(((Image)sender)).Y / ((Image)sender).ActualHeight * (double)imageH);
                int z = (int)Z_slider.Value;
                segmentSelectionXY(x, y, z);
                //segmentSelection(x, y, z);
            }
        }

        private void ortho_zx_MouseAction(object sender, System.Windows.Input.MouseEventArgs e)
        {
            if (Keyboard.IsKeyDown(Key.LeftShift) || Keyboard.IsKeyDown(Key.RightShift)) return;
            if (e.LeftButton.Equals(MouseButtonState.Pressed))
            {
                Z_slider.Value = (int)(e.GetPosition(((Image)sender)).Y / ((Image)sender).ActualHeight * (double)imageZ);
                X_slider.Value = (int)(e.GetPosition(((Image)sender)).X / ((Image)sender).ActualWidth * (double)imageW);
            }
            if (e.MiddleButton.Equals(MouseButtonState.Pressed))
            {
                int x = (int)(e.GetPosition(((Image)sender)).X / ((Image)sender).ActualWidth * (double)imageW);
                int y = (int)Y_slider.Value;
                int z = (int)(e.GetPosition(((Image)sender)).Y / ((Image)sender).ActualHeight * (double)imageZ);
                segmentSelection(x, y, z);
            }
        }
       
        private void ortho_yz_MouseAction(object sender, System.Windows.Input.MouseEventArgs e)
        {
            if (Keyboard.IsKeyDown(Key.LeftShift) || Keyboard.IsKeyDown(Key.RightShift)) return;
            if (e.LeftButton.Equals(MouseButtonState.Pressed))
            {
                Y_slider.Value = (int)(e.GetPosition(((Image)sender)).Y / ((Image)sender).ActualHeight * (double)imageH);
                Z_slider.Value = (int)(e.GetPosition(((Image)sender)).X / ((Image)sender).ActualWidth * (double)imageZ);
            }
            if (e.MiddleButton.Equals(MouseButtonState.Pressed))
            {
                int x = (int)X_slider.Value;
                int y = (int)(e.GetPosition(((Image)sender)).Y / ((Image)sender).ActualHeight * (double)imageH);
                int z = (int)(e.GetPosition(((Image)sender)).X / ((Image)sender).ActualWidth * (double)imageZ);
                segmentSelection(x, y, z);
            }
        }



        private void SM_apply_button_Click(object sender, RoutedEventArgs e)
        {
            c1.smooth3D((int)SM_block_size_slider.Value, SM_mean_rb.IsChecked.Value ? 1/*mean*/ : 0/*gaussian*/, true);
            updateImg();
        }
		
		private void DL_apply_button_Click(object sender, RoutedEventArgs e)
		{
			c1.dilation3D((int)DL_radius_slider.Value, DL_sphere_rb.IsChecked.Value ? Class1.C1_FLT_SPHERE : Class1.C1_FLT_CUBE, true);
			updateImg();
		}

		private void ER_apply_button_Click(object sender, RoutedEventArgs e)
		{
			c1.erosion3D((int)ER_radius_slider.Value, ER_sphere_rb.IsChecked.Value ? Class1.C1_FLT_SPHERE : Class1.C1_FLT_CUBE, true);
			updateImg();
		}

		private void AT_apply_button_Click(object sender, RoutedEventArgs e)
        {
			//親のListBoxItemを取得
			DependencyObject parent = LogicalTreeHelper.GetParent((DependencyObject)sender);
			while (!(parent is ListBoxItem))
			{
				parent = LogicalTreeHelper.GetParent(parent);
				if (parent == null) return;
			}
			
			var paramList = new Dictionary<String, DependencyObject>();

			LogicalTreeElementList.EnumerateVisualChildren(parent, GetParams, paramList);
            
			//Console.WriteLine(string.Format("FilterType: {0}",	((RadioButton)paramList["AT_mean_rb"]).IsChecked.Value ? 1/*mean*/ : 0/*gaussian*/));
			//Console.WriteLine(string.Format("BlockSize: {0}",	((Slider)paramList["AT_block_size_slider"]).Value));
			//Console.WriteLine(string.Format("C: {0}",			((Slider)paramList["AT_C_slider"]).Value));
            
			c1.AdaptiveThreshold3D((int)((Slider)paramList["AT_block_size_slider"]).Value,
								   (float)((Slider)paramList["AT_C_slider"]).Value * -1,
								   ((RadioButton)paramList["AT_mean_rb"]).IsChecked.Value ? 1/*mean*/ : 0/*gaussian*/,
								   true );
			updateImg();
        }

		private void AT_applyXY_button_Click(object sender, RoutedEventArgs e)
		{
			//親のListBoxItemを取得
			DependencyObject parent = LogicalTreeHelper.GetParent((DependencyObject)sender);
			while (!(parent is ListBoxItem))
			{
				parent = LogicalTreeHelper.GetParent(parent);
				if (parent == null) return;
			}

			var paramList = new Dictionary<String, DependencyObject>();

			LogicalTreeElementList.EnumerateVisualChildren(parent, GetParams, paramList);

			//Console.WriteLine(string.Format("FilterType: {0}",	((RadioButton)paramList["AT_mean_rb"]).IsChecked.Value ? 1/*mean*/ : 0/*gaussian*/));
			//Console.WriteLine(string.Format("BlockSize: {0}",	((Slider)paramList["AT_block_size_slider"]).Value));
			//Console.WriteLine(string.Format("C: {0}",			((Slider)paramList["AT_C_slider"]).Value));

			c1.AdaptiveThreshold2D((int)((Slider)paramList["AT_block_size_slider"]).Value,
                                   (float)((Slider)paramList["AT_C_slider"]).Value * -1,
								   ((RadioButton)paramList["AT_mean_rb"]).IsChecked.Value ? 1/*mean*/ : 0/*gaussian*/,
								   true );
			updateImg();
		}


		private void ADTHLK_apply_button_Click(object sender, RoutedEventArgs e)
		{
			//親のListBoxItemを取得
			DependencyObject parent = LogicalTreeHelper.GetParent((DependencyObject)sender);
			while (!(parent is ListBoxItem))
			{
				parent = LogicalTreeHelper.GetParent(parent);
				if (parent == null) return;
			}

			var paramList = new Dictionary<String, DependencyObject>();

			LogicalTreeElementList.EnumerateVisualChildren(parent, GetParams, paramList);
            
			//Console.WriteLine(string.Format("FilterType: {0}",	 ((RadioButton)paramList["ADTHLK_mean_rb"]).IsChecked.Value ? 1/*mean*/ : 0/*gaussian*/));
			//Console.WriteLine(string.Format("BlockSize: {0}",	 ((Slider)paramList["ADTHLK_block_size_slider"]).Value));
			//Console.WriteLine(string.Format("Level: {0}",		 ((Slider)paramList["ADTHLK_lv_slider"]).Value));
			//Console.WriteLine(string.Format("Z_Correction: {0}", ((Slider)paramList["ADTHLK_Z_Correction_slider"]).Value));
			//Console.WriteLine(string.Format("C: {0}",			 ((Slider)paramList["ADTHLK_C_slider"]).Value));

			c1.AdaptiveThreshold3DLineKernels((int)((Slider)paramList["ADTHLK_block_size_slider"]).Value,
											  (int)((Slider)paramList["ADTHLK_lv_slider"]).Value,
											  (float)((Slider)paramList["ADTHLK_Z_Correction_slider"]).Value,
											  (float)((Slider)paramList["ADTHLK_C_slider"]).Value * -1,
											  ((RadioButton)paramList["ADTHLK_mean_rb"]).IsChecked.Value ? 1/*mean*/ : 0/*gaussian*/);
			updateImg();
		}
		
		private void SGL_apply_button_Click(object sender, RoutedEventArgs e)
		{
			//親のListBoxItemを取得
			DependencyObject parent = LogicalTreeHelper.GetParent((DependencyObject)sender);
			while (!(parent is ListBoxItem))
			{
				parent = LogicalTreeHelper.GetParent(parent);
				if (parent == null) return;
			}

			var paramList = new Dictionary<String, DependencyObject>();

			LogicalTreeElementList.EnumerateVisualChildren(parent, GetParams, paramList);

			/*
			foreach (KeyValuePair<string, DependencyObject> pair in paramList)
			{
				Console.WriteLine(string.Format("Key : {0} \n/ Value : {1}", pair.Key, pair.Value));
			}
			*/

			//Console.WriteLine(string.Format("FilterType: {0}", ((RadioButton)paramList["SGL_mean_rb"]).IsChecked.Value ? 1/*mean*/ : 0/*gaussian*/));
			//Console.WriteLine(string.Format("BlockSize: {0}", ((Slider)paramList["SGL_block_size_slider"]).Value));
			//Console.WriteLine(string.Format("Level: {0}", ((Slider)paramList["SGL_lv_slider"]).Value));
			//Console.WriteLine(string.Format("Z_Correction: {0}", ((Slider)paramList["SGL_Z_Correction_slider"]).Value));
			//Console.WriteLine(string.Format("minC: {0}", ((Slider)paramList["SGL_minC_slider"]).Value));
			//Console.WriteLine(string.Format("maxC: {0}", ((Slider)paramList["SGL_maxC_slider"]).Value));
			//Console.WriteLine(string.Format("Interval: {0}", ((Slider)paramList["SGL_interval_slider"]).Value));
			//Console.WriteLine(string.Format("minVol: {0}", ((Slider)paramList["SGL_minVol_slider"]).Value));
			//Console.WriteLine(string.Format("Tolerance: {0}", ((Slider)paramList["SGL_minInvalidStructureArea_slider"]).Value));
			//Console.WriteLine(string.Format("Closing: {0}", ((Slider)paramList["SGL_Closing_slider"]).Value));

			c1.segmentation_SobelLikeADTH((int)((Slider)paramList["SGL_block_size_slider"]).Value,
										  (int)((Slider)paramList["SGL_lv_slider"]).Value,
										  (float)((Slider)paramList["SGL_Z_Correction_slider"]).Value,
										  (float)((Slider)paramList["SGL_minC_slider"]).Value * -1,
										  (float)((Slider)paramList["SGL_maxC_slider"]).Value * -1,
										  (float)((Slider)paramList["SGL_interval_slider"]).Value * -1,
										  ((RadioButton)paramList["SGL_mean_rb"]).IsChecked.Value ? 1/*mean*/ : 0/*gaussian*/,
										  (int)((Slider)paramList["SGL_minVol_slider"]).Value,
										  (int)((Slider)paramList["SGL_minInvalidStructureArea_slider"]).Value,
										  (int)((Slider)paramList["SGL_Closing_slider"]).Value);

			updateImg();
		}

		private void HMT_apply_button_Click(object sender, RoutedEventArgs e)
		{
			//親のListBoxItemを取得
			DependencyObject parent = LogicalTreeHelper.GetParent((DependencyObject)sender);
			while (!(parent is ListBoxItem))
			{
				parent = LogicalTreeHelper.GetParent(parent);
				if (parent == null) return;
			}

			var paramList = new Dictionary<String, DependencyObject>();

			LogicalTreeElementList.EnumerateVisualChildren(parent, GetParams, paramList);

			Console.WriteLine(string.Format("h: {0}", ((Slider)paramList["HMT_h_slider"]).Value));
			Console.WriteLine(string.Format("Check: {0}", ((Slider)paramList["HMT_chkInterval_slider"]).Value));

			c1.hMinimaTransform3D((float)((Slider)paramList["HMT_h_slider"]).Value,
								  (int)((Slider)paramList["HMT_chkInterval_slider"]).Value,
								  true);
			updateImg();
		}

		private void SGH_apply_button_Click(object sender, RoutedEventArgs e)
		{
			//親のListBoxItemを取得
			DependencyObject parent = LogicalTreeHelper.GetParent((DependencyObject)sender);
			while (!(parent is ListBoxItem))
			{
				parent = LogicalTreeHelper.GetParent(parent);
				if (parent == null) return;
			}

			var paramList = new Dictionary<String, DependencyObject>();

			LogicalTreeElementList.EnumerateVisualChildren(parent, GetParams, paramList);

			/*
			foreach (KeyValuePair<string, DependencyObject> pair in paramList)
			{
				Console.WriteLine(string.Format("Key : {0} \n/ Value : {1}", pair.Key, pair.Value));
			}
			*/

			Console.WriteLine(string.Format("minh: {0}", ((Slider)paramList["SGH_minh_slider"]).Value));
			Console.WriteLine(string.Format("maxh: {0}", ((Slider)paramList["SGH_maxh_slider"]).Value));
			//Console.WriteLine(string.Format("check: {0}", ((Slider)paramList["SGH_check_slider"]).Value));
			Console.WriteLine(string.Format("Interval: {0}", ((Slider)paramList["SGH_interval_slider"]).Value));
			Console.WriteLine(string.Format("minVol: {0}", ((Slider)paramList["SGH_minVol_slider"]).Value));
			Console.WriteLine(string.Format("Tolerance: {0}", ((Slider)paramList["SGH_minInvalidStructureArea_slider"]).Value));
			Console.WriteLine(string.Format("Closing: {0}", ((Slider)paramList["SGH_Closing_slider"]).Value));

			c1.segmentation_hMinimaTransform((float)((Slider)paramList["SGH_minh_slider"]).Value,
											 (float)((Slider)paramList["SGH_maxh_slider"]).Value,
											 (float)((Slider)paramList["SGH_interval_slider"]).Value,
											 (int)((Slider)paramList["SGH_minVol_slider"]).Value,
											 (int)((Slider)paramList["SGH_minInvalidStructureArea_slider"]).Value,
											 (int)((Slider)paramList["SGH_Closing_slider"]).Value);

			updateImg();
		}

        private void SGT_apply_button_Click(object sender, RoutedEventArgs e)
        {
            c1.segmentation_Thresholding((float)SGT_minth_slider.Value,
                                         (float)SGT_maxth_slider.Value,
                                         (float)SGT_interval_slider.Value,
                                         (int)SGT_minVol_slider.Value,
                                         (int)SGT_minInvalidStructureArea_slider.Value,
                                         (int)SGT_Closing_slider.Value);
            updateImg();
        }

		private void FF_apply_button_Click(object sender, RoutedEventArgs e)
		{
			int connectType = Class1.C1_SEG_CONNECT6;

			if(FF_connectType6_rb.IsChecked.Value) connectType = Class1.C1_SEG_CONNECT6;
			if(FF_connectType18_rb.IsChecked.Value)connectType = Class1.C1_SEG_CONNECT18;
			if(FF_connectType26_rb.IsChecked.Value)connectType = Class1.C1_SEG_CONNECT26;

			if (FF_validation_cb.IsChecked.Value)
			{
				c1.segmentation_FloodFill((float)FF_th_slider.Value,
										  connectType,
										  (int)FF_minSize_slider.Value,
										  (int)FF_noiseRad_slider.Value,
										  (int)FF_wallRad_slider.Value,
										  (int)FF_minInvalidStructureArea_slider.Value * (int)FF_wallRad_slider.Value,
										  (bool)FF_saveValidSeg_cb.IsChecked.Value,
										  (bool)FF_saveInvalidSeg_cb.IsChecked.Value);

			}
			else
			{
				c1.segmentation_FloodFill((float)FF_th_slider.Value, connectType, (int)FF_minSize_slider.Value);
			}

			updateImg();
		}

        private void H_generate_button_Click(object sender, RoutedEventArgs e)
        {
            if (c1.empty()) return;

            if (H_gauss_rb.IsChecked.Value) c1.generateHeightMap((int)H_block_size_slider.Value, (int)H_Z_slider.Value, (float)H_th_slider.Value, 0, (int)H_SmLv_slider.Value);
            if (H_mean_rb.IsChecked.Value) c1.generateHeightMap((int)H_block_size_slider.Value, (int)H_Z_slider.Value, (float)H_th_slider.Value, 1, (int)H_SmLv_slider.Value);


            updateImg();
            
			if (H_show_cb.IsChecked.Value)
            {
                //sbw.Close();
                //sbw = new SubWindow();

                IntPtr data;
                int byteperpixel, width, height;
                if (c1.getHeightMapDataArrayXY(out data, out byteperpixel, out width, out height) == 0) return;
                sbw.HeightMap.Source = BitmapSource.Create(width, height, 96, 96, PixelFormats.Rgb24, null, data, width * height * byteperpixel, width * byteperpixel);
                sbw.HeightMap.Stretch = Stretch.None;
                sbw.Show();
                sbw.SizeToContent = SizeToContent.WidthAndHeight;
                sbw.SizeToContent = SizeToContent.Manual;
                sbw.HeightMap.Stretch = Stretch.Uniform;
            }
        }


		private void H_clear_button_Click(object sender, RoutedEventArgs e)
		{
			if (c1.empty()) return;
			c1.inactivateHeightMap();
			updateImg();
		}

        private void SaveDest2DImage_Executed(object sender, ExecutedRoutedEventArgs e)
        {
            Console.WriteLine("COMMAND");
        }

        private void H_projParamsChanged(object sender, RoutedEventArgs e)
        {
            if (H_hproj_cb == null || H_show_cb == null || H_normal_rb == null || H_offset_slider == null || H_range_slider == null || H_depth_slider == null || H_proj_th_slider == null) return;

            c1.setHmapActivity(H_hproj_cb.IsChecked.Value);
            c1.setHmapVisibility(H_show_cb.IsChecked.Value);
            if (H_normal_rb.IsChecked.Value) c1.setProjectionMode(0);
            else c1.setProjectionMode(1);
            c1.setHmapOffset((float)H_offset_slider.Value);
            c1.setHmapProjRange((float)H_range_slider.Value);
            c1.setHmapProjDepth((float)H_depth_slider.Value);
            c1.setProjectionThreshold((float)H_proj_th_slider.Value);
            updateImg();
        }

        private void DC_range_slider_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            c1.setDmapRange((float)DC_range_slider.Value);
            updateImg();
        }

        private void DC_apply_button_Click(object sender, RoutedEventArgs e)
        {
            c1.generateDepthMap();
        }

        private void DC_cb_CheckChanged(object sender, RoutedEventArgs e)
		{
			updateImg();
		}

        private void DC_coefficient_slider_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
			if (!DC_coefficient_slider.IsEnabled) return;
			c1.setDmapCoefficient((float)DC_coefficient_slider.Value);
            updateImg();
        }

        private void DC_order_slider_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
			if (!DC_order_slider.IsEnabled) return;
            c1.setDmapOrder((float)DC_order_slider.Value);
            updateImg();
        }

        private void BC_max_slider_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            c1.setBCmax((float)BC_max_slider.Value);
            if (BC_min_slider != null) if (BC_min_slider.Value > BC_max_slider.Value) BC_min_slider.Value = BC_max_slider.Value;
            updateImg();
        }

        private void BC_min_slider_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            c1.setBCmin((float)BC_min_slider.Value);
            if (BC_max_slider != null) if (BC_min_slider.Value > BC_max_slider.Value) BC_max_slider.Value = BC_min_slider.Value;
            updateImg();
        }

        private void BC_apply_button_Click(object sender, RoutedEventArgs e)
        {
            c1.applyBlightnessContrast();
            DC_cb.IsChecked = false;
            BC_max_slider.Value = 1.0;
            BC_min_slider.Value = 0.0;
            updateImg();
        }

		private void ZBC_slider_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
		{
			if (ZBC_ch_slider != null && ZBC_coefficient_slider != null && ZBC_order_slider != null)
			{
				c1.setZBCparams((int)ZBC_ch_slider.Value, (float)ZBC_coefficient_slider.Value, (float)ZBC_order_slider.Value);
				updateImg();
			}
		}

		///<summary>
		///可視化するItemが可視化されているItemの最後尾へくるように並び替えも行います
		///</summary>
		private void VisualizeListBoxItem(ListBoxItem item)
		{
			if (item.Visibility == System.Windows.Visibility.Visible) return;

			var parent = LogicalTreeHelper.GetParent((DependencyObject)item);
			if (parent is ListBox)
			{
				var lb = (ListBox)parent;
				int index;
				for (index = 0; index < lb.Items.Count; index++)
				{
					var cur = lb.ItemContainerGenerator.ContainerFromIndex(index) as ListBoxItem;
					if (cur.Visibility == System.Windows.Visibility.Collapsed) break;
				}
				//itemの位置はindexより後ろにあるのでRemoveしても挿入すべき位置はずれない
				lb.Items.Remove(item);
				lb.Items.Insert(index, item);

				item.Visibility = System.Windows.Visibility.Visible;
			}
			
		}

		private void HideListBoxItem(ListBoxItem item)
		{
			if (item.Visibility == System.Windows.Visibility.Collapsed) return;

			var parent = LogicalTreeHelper.GetParent((DependencyObject)item);
			if (parent is ListBox)
			{
				var lb = (ListBox)parent;
				lb.Items.Remove(item);
				lb.Items.Add(item);
				item.Visibility = System.Windows.Visibility.Collapsed;
			}

		}

		private void FilterPanelCloseButton_Click(object sender, RoutedEventArgs e)
		{
			//親のListBoxItemを取得
			DependencyObject parent = LogicalTreeHelper.GetParent((DependencyObject)sender);
			while (!(parent is ListBoxItem))
			{
				parent = LogicalTreeHelper.GetParent(parent);
				if (parent == null) return;
			}

			HideListBoxItem((ListBoxItem)parent);
		}

		private void cbi_Selected(object sender, RoutedEventArgs e)
		{
			var cbi = (ComboBoxItem)sender;
			
			switch (cbi.Name)
			{
				case "cbi_Smoothing":
					VisualizeListBoxItem(lbi_Smoothing);
					break;
                case "cbi_Threshold":
                    VisualizeListBoxItem(lbi_Threshold);
                    break;
				case "cbi_Dilation":
					VisualizeListBoxItem(lbi_Dilation);
					break;
				case "cbi_Erosion":
					VisualizeListBoxItem(lbi_Erosion);
					break;
                case "cbi_LoacalTH":
                    VisualizeListBoxItem(lbi_LoacalTH);
					break;
                case "cbi_DSLT":
					VisualizeListBoxItem(lbi_DSLT);
					break;
				case "cbi_hMinima":
					VisualizeListBoxItem(lbi_H_minima_Transform);
					break;
				case "cbi_segDSLT":
                    VisualizeListBoxItem(lbi_Segmentation_DSLT);
					break;
				case "cbi_segHMT":
					VisualizeListBoxItem(lbi_Segmentation_H_minima);
					break;
                case "cbi_segTH":
                    VisualizeListBoxItem(lbi_Segmentation_TH);
                    break;
				case "cbi_segFF":
					VisualizeListBoxItem(lbi_Segmentation_FloodFill);
					break;
                case "cbi_SS_th":
					VisualizeListBoxItem(lbi_SegmentSelection_th);
                    break;
                case "cbi_segD":
                    VisualizeListBoxItem(lbi_SegmentDilation);
                    break;
                case "cbi_segE":
                    VisualizeListBoxItem(lbi_SegmentErosion);
                    break;
/*              case "cbi_segC":
                    VisualizeListBoxItem(lbi_SegmentClosing);
                    break;
                case "cbi_segO":
                    VisualizeListBoxItem(lbi_SegmentOpening);
                    break;
                case "cbi_re_seg":
                    VisualizeListBoxItem(lbi_Re_Segmention_DSLT);
                    break;
*/              case "cbi_watershed":
                    VisualizeListBoxItem(lbi_Watershed);
                    break;
			}

			listBox.Items.Refresh();

			//Console.WriteLine(cbi.Name);
		}

		private void cmb_SelectionChanged(object sender, SelectionChangedEventArgs e)
		{
			((ComboBox)sender).SelectedIndex = 0;
		}


		private void GetParams(DependencyObject target, int level, Object option)
		{
			var paramList = (Dictionary<String, DependencyObject>)option;
			if (target is CheckBox || target is RadioButton || target is Slider)
				paramList.Add((String)((Control)target).Name, target);
		}

		private UIElement WPF_DeepCopy(UIElement element)
		{
			string shapestring = System.Windows.Markup.XamlWriter.Save(element);
			Console.WriteLine(shapestring);
			var sr = new System.IO.StringReader(shapestring);
			var xtr = new System.Xml.XmlTextReader(sr);
			var deepcopy = (UIElement)System.Windows.Markup.XamlReader.Load(xtr);
			return deepcopy;
		}

		private int getCurrentScalingType(){
			if (SC_none_rb.IsChecked.Value)		return Class1.C1_SC_NONE;
			if (SC_areaAve_rb.IsChecked.Value)	return Class1.C1_SC_AREA_AVE;
			if (SC_lanczos2_rb.IsChecked.Value)	return Class1.C1_SC_LANCZOS2;
			if (SC_lanczos3_rb.IsChecked.Value)	return Class1.C1_SC_LANCZOS3;

			return Class1.C1_SC_NONE;
		}

		private int getCurrentChannel()
		{
			return (int)ch_slider.Value;
		}

		private void setZScalingType()
		{
			if (c1.empty()) return;

			c1.setScalingType(getCurrentScalingType());
			
			init();

			updateImg();
		}

		private void setChannel()
		{
			if (c1.empty()) return;

			c1.setChannel(getCurrentChannel());

			updateImg();
		}

		private void SC_rb_Checked(object sender, RoutedEventArgs e)
		{
			setZScalingType();
		}

		private void Ch_slider_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
		{
			setChannel();
		}

		private void applyChanges_button_Click(object sender, RoutedEventArgs e)
		{
			c1.applyChanges();
			BC_apply_button_Click(sender, e);
		}

		private void SG_min_size_slider_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
		{
			c1.setSegMinVol((int)SG_min_size_slider.Value);
			updateImg();
		}

		private void SG_src_cb_Checked(object sender, RoutedEventArgs e)
		{
			c1.setSrcSegVisibility((bool)SG_src_cb.IsChecked.Value);
			updateImg();
		}

		private void SG_src_cb_Unchecked(object sender, RoutedEventArgs e)
		{
			c1.setSrcSegVisibility((bool)SG_src_cb.IsChecked.Value);
			updateImg();
		}

        private void SG_dst_cb_Checked(object sender, RoutedEventArgs e)
        {
            c1.setDstSegVisibility((bool)SG_dst_cb.IsChecked.Value);
            updateImg();
        }

        private void SG_dst_cb_Unchecked(object sender, RoutedEventArgs e)
        {
            c1.setDstSegVisibility((bool)SG_dst_cb.IsChecked.Value);
            updateImg();
        }

        private void SST_apply_button_Click(object sender, RoutedEventArgs e)
        {
            if(!SST_deselect_cb.IsChecked.Value)c1.selectSegments_TH((float)SST_th_slider.Value);
            else c1.deselectSegments_TH((float)SST_th_slider.Value);
            updateImg();
        }
        
        private void SGD_apply_button_Click(object sender, RoutedEventArgs e)
        {
            c1.dilateSphereSelectedSegments((int)SGD_r_slider.Value);
            updateImg();
        }

        private void SGE_apply_button_Click(object sender, RoutedEventArgs e)
        {
            c1.erodeSphereSelectedSegments((int)SGE_r_slider.Value, SGE_clamp_cb.IsChecked.Value);
            updateImg();
        }

        private void SGC_apply_button_Click(object sender, RoutedEventArgs e)
        {
        
        }

        private void SGO_apply_button_Click(object sender, RoutedEventArgs e)
        {

        }

        private void RSD_apply_button_Click(object sender, RoutedEventArgs e)
        {

        }

        private void H_crop_ValueChanged(object sender, RoutedEventArgs e)
        {
            if (H_upper_slider == null || H_lower_slider == null || H_crop_cb == null || H_crop_hmap_cb == null || H_halo_slider == null) return;

            if (H_upper_slider.Value > H_lower_slider.Value)
            {
                if (sender == H_upper_slider) H_lower_slider.Value = H_upper_slider.Value;
                else H_upper_slider.Value = H_lower_slider.Value;
            }
            //set crop parameters to c1
            c1.setCroppingParams(H_crop_cb.IsChecked.Value, H_crop_hmap_cb.IsChecked.Value, (int)H_upper_slider.Value, (int)H_lower_slider.Value, (int)H_halo_slider.Value);
        }

        private void H_crop_apply_button_Click(object sender, RoutedEventArgs e)
        {
            c1.cropSegments();
            updateImg();
        }

        private void WS_apply_button_Click(object sender, RoutedEventArgs e)
        {
            c1.watershed3D((float)WS_stride_slider.Value);
            //WS_stride_slider.Value += 0.001f;
            updateImg();
        }

        private void TH_apply_button_Click(object sender, RoutedEventArgs e)
        {
            c1.thresholding((float)TH_th_slider.Value);
            updateImg();
        }

        private void MenuItem_Reset_Click(object sender, RoutedEventArgs e)
        {
            ortho_view.ColumnDefinitions.ElementAt(0).Width = new GridLength(1.0, GridUnitType.Star);
            ortho_view.ColumnDefinitions.ElementAt(2).Width = new GridLength(1.0, GridUnitType.Star);
        }

        private void ortho_view_SizeChanged(object sender, SizeChangedEventArgs e)
        {
            GridLength width = ortho_view.ColumnDefinitions.ElementAt(0).Width;
        }

        private void view_SizeChanged(object sender, SizeChangedEventArgs e)
        {
            viewSizeChange(sender);
        }

        private void viewSizeChange(object target)
        {
            if (imageW <= 0 || imageH <= 0 || imageZ <= 0) return;

            var parent = LogicalTreeHelper.GetParent((DependencyObject)target);
            if (parent is Grid)
            {
                Grid root = (Grid)parent;
                int row = Grid.GetRow((UIElement)target);
                int col = Grid.GetColumn((UIElement)target);
                double panelW = ((FrameworkElement)target).ActualWidth;
                double panelH = ((FrameworkElement)target).ActualHeight;

                var items = root.Children.Cast<UIElement>().Where(i => Grid.GetRow(i) == row &&
                                                                       Grid.GetColumn(i) == col &&
                                                                       i != (UIElement)target &&
                                                                       i is Grid);
                if (items.Count() == 0) return;
                Grid view = (Grid)items.ElementAt(0);
                

                var svs = view.Children.Cast<UIElement>().Where(i => Grid.GetRow(i) == 0 && Grid.GetColumn(i) == 0 && i is ScrollViewer);
                if (svs.Count() == 0) return;
                ScrollViewer sv_xy = (ScrollViewer)svs.ElementAt(0);
                var cont = sv_xy.Content;
                if (cont == null) return;
                if (!(cont is Grid)) return;
                Grid xypanel = (Grid)cont;

                svs = view.Children.Cast<UIElement>().Where(i => Grid.GetRow(i) == 2 && Grid.GetColumn(i) == 0 && i is ScrollViewer);
                if (svs.Count() == 0) return;
                ScrollViewer sv_zx = (ScrollViewer)svs.ElementAt(0);
                cont = sv_zx.Content;
                if (cont == null) return;
                if (!(cont is Grid)) return;
                Grid zxpanel = (Grid)cont;

                svs = view.Children.Cast<UIElement>().Where(i => Grid.GetRow(i) == 0 && Grid.GetColumn(i) == 2 && i is ScrollViewer);
                if (svs.Count() == 0) return;
                ScrollViewer sv_yz = (ScrollViewer)svs.ElementAt(0);
                cont = sv_yz.Content;
                if (cont == null) return;
                if (!(cont is Grid)) return;
                Grid yzpanel = (Grid)cont;

                if (xypanel.ActualWidth <= 0.0 || yzpanel.ActualHeight <= 0.0 || zxpanel.ActualHeight <= 0.0) return;

                double inner_grid_aspect_ratio = (18.0 + (double)imageW + 3.0 + (double)imageZ) / (18.0 + (double)imageH + 3.0 + (double)imageZ);
                double outer_panel_aspect_ratio = panelW / panelH;

                double offsetx = sv_xy.HorizontalOffset / xypanel.ActualWidth;
                double offsety = sv_yz.VerticalOffset / yzpanel.ActualHeight;
                double offsetz = sv_zx.VerticalOffset / zxpanel.ActualHeight;

                double w, h, d;
                if (inner_grid_aspect_ratio > outer_panel_aspect_ratio)
                {
                    w = (panelW - 3.0 - 18.0) * (double)imageW / (double)(imageW + imageZ) - 12.0/*margin*/;
                    w = (w > sv_xy.MinWidth) ? w : sv_xy.MinWidth;

                    h = w * (view_ratioH / view_ratioW);
                    d = w * (view_ratioD / view_ratioW);
                }
                else
                {
                    h = (panelH - 3.0 - 18.0) * (double)imageH / (double)(imageH + imageZ) - 12.0/*margin*/;
                    h = (h > sv_xy.MinHeight) ? h : sv_xy.MinHeight;

                    w = h * (view_ratioW / view_ratioH);
                    d = h * (view_ratioD / view_ratioH);
                }

                double im_mag = calcMag(im_mag_lv);

                sv_xy.Width = w + 18.0;
                sv_xy.Height = h + 18.0;
                xypanel.Width = w * sv_im_ratioW * im_mag;
                xypanel.Height = h * sv_im_ratioH * im_mag;

                sv_zx.Width = w + 18.0;
                sv_zx.Height = d;
                zxpanel.Width = w * sv_im_ratioW * im_mag;
                zxpanel.Height = d * sv_im_ratioD * im_mag;

                sv_yz.Width = d;
                sv_yz.Height = h + 18.0;
                yzpanel.Width = d * sv_im_ratioD * im_mag;
                yzpanel.Height = h * sv_im_ratioH * im_mag;

                if (sv_xy == dst_sv_xy)
                {
                    debug_cmp_txt.Text = xypanel.ActualWidth.ToString();
                    debug_txt.Text = (w * sv_im_ratioW * im_mag).ToString();
                }

                sv_xy.UpdateLayout();
                sv_yz.UpdateLayout();
                sv_zx.UpdateLayout();

                sv_xy.ScrollToHorizontalOffset(offsetx * w * sv_im_ratioW * im_mag);
                sv_xy.ScrollToVerticalOffset(offsety * h * sv_im_ratioH * im_mag);
                sv_yz.ScrollToHorizontalOffset(offsetz * d * sv_im_ratioD * im_mag);
                sv_yz.ScrollToVerticalOffset(offsety * h * sv_im_ratioH * im_mag);
                sv_zx.ScrollToHorizontalOffset(offsetx * w * sv_im_ratioW * im_mag);
                sv_zx.ScrollToVerticalOffset(offsetz * d * sv_im_ratioD * im_mag);

            }
        }

        private void view_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            if (Keyboard.IsKeyDown(Key.LeftShift) || Keyboard.IsKeyDown(Key.RightShift))
            {
                im_mag_lv++;

                view_magnification(calcMag(im_mag_lv) / calcMag(im_mag_lv - 1), e);
            }
        }

        private void view_MouseRightButtonDown(object sender, MouseButtonEventArgs e)
        {
            if (Keyboard.IsKeyDown(Key.LeftShift) || Keyboard.IsKeyDown(Key.RightShift))
            {
                im_mag_lv--;
                
                if (im_mag_lv < 0) im_mag_lv = 0;
                else view_magnification(calcMag(im_mag_lv) / calcMag(im_mag_lv + 1), e);
            }
        }

        private double calcMag(int lv)
        {
            if(lv <= 0) return 1.0;
            
            int m = lv / 2;
            int mag_p = 100;
            
            mag_p += 50 * ((1 << m) - 1);

            if (lv % 2 == 1) m++;
            mag_p += 50 * ((1 << m) - 1);

            return (double)mag_p / 100.0;
        }

        private void view_magnification(double rel_mag, MouseButtonEventArgs e)
        {
            double? x, y, z, tmpx, tmpy, tmpz;

            x = null;
            y = null;
            z = null;

            if (src_sv_xy.IsMouseOver)
            {
                tmpx = e.GetPosition(ortho_xy).X / ortho_xy.ActualWidth;
                tmpy = e.GetPosition(ortho_xy).Y / ortho_xy.ActualHeight;
                if (tmpx >= 0.0 && tmpy >= 0.0 && tmpx <= 1.0 && tmpy <= 1.0)
                {
                    x = tmpx;
                    y = tmpy;
                }
            }
            else if (src_sv_zx.IsMouseOver)
            {
                tmpx = e.GetPosition(ortho_zx).X / ortho_zx.ActualWidth;
                tmpz = e.GetPosition(ortho_zx).Y / ortho_zx.ActualHeight;
                if (tmpx >= 0.0 && tmpz >= 0.0 && tmpx <= 1.0 && tmpz <= 1.0)
                {
                    x = tmpx;
                    z = tmpz;
                }
            }
            else if (src_sv_yz.IsMouseOver)
            {
                tmpz = e.GetPosition(ortho_yz).X / ortho_yz.ActualWidth;
                tmpy = e.GetPosition(ortho_yz).Y / ortho_yz.ActualHeight;
                if (tmpz >= 0.0 && tmpy >= 0.0 && tmpz <= 1.0 && tmpy <= 1.0)
                {
                    z = tmpz;
                    y = tmpy;
                }
            }
            else if (dst_sv_xy.IsMouseOver)
            {
                tmpx = e.GetPosition(dst_ortho_xy).X / dst_ortho_xy.ActualWidth;
                tmpy = e.GetPosition(dst_ortho_xy).Y / dst_ortho_xy.ActualHeight;
                if (tmpx >= 0.0 && tmpy >= 0.0 && tmpx <= 1.0 && tmpy <= 1.0)
                {
                    x = tmpx;
                    y = tmpy;
                }
            }
            else if (dst_sv_zx.IsMouseOver)
            {
                tmpx = e.GetPosition(dst_ortho_zx).X / dst_ortho_zx.ActualWidth;
                tmpz = e.GetPosition(dst_ortho_zx).Y / dst_ortho_zx.ActualHeight;
                if (tmpx >= 0.0 && tmpz >= 0.0 && tmpx <= 1.0 && tmpz <= 1.0)
                {
                    x = tmpx;
                    z = tmpz;
                }
            }
            else if (dst_sv_yz.IsMouseOver)
            {
                tmpz = e.GetPosition(dst_ortho_yz).X / dst_ortho_yz.ActualWidth;
                tmpy = e.GetPosition(dst_ortho_yz).Y / dst_ortho_yz.ActualHeight;
                if (tmpz >= 0.0 && tmpy >= 0.0 && tmpz <= 1.0 && tmpy <= 1.0)
                {
                    z = tmpz;
                    y = tmpy;
                }
            }

            if (x == null) x = X_slider.Value / imageW;
            if (y == null) y = Y_slider.Value / imageH;
            if (z == null) z = Z_slider.Value / imageZ;

            double src_x = (double)x, src_y = (double)y, src_z = (double)z;
            double dst_x = (double)x, dst_y = (double)y, dst_z = (double)z;


            src_x *= ortho_xy.ActualWidth * rel_mag;
            src_y *= ortho_yz.ActualHeight * rel_mag;
            src_z *= ortho_zx.ActualHeight * rel_mag;

            double xmax = ortho_xy.ActualWidth * rel_mag - src_sv_xy.ViewportWidth;
            double ymax = ortho_yz.ActualHeight * rel_mag - src_sv_yz.ViewportHeight;
            double zmax = ortho_zx.ActualHeight * rel_mag - src_sv_zx.ViewportHeight;

            src_x -= src_sv_xy.ViewportWidth / 2;
            if (src_x < 0.0) src_x = 0.0;
            if (src_x > xmax) src_x = xmax;

            src_y -= src_sv_yz.ViewportHeight / 2;
            if (src_y < 0.0) src_y = 0.0;
            if (src_y > ymax) src_y = ymax;

            src_z -= src_sv_zx.ViewportHeight / 2;
            if (src_z < 0.0) src_z = 0.0;
            if (src_z > zmax) src_z = zmax;


            dst_x *= dst_ortho_xy.ActualWidth * rel_mag;
            dst_y *= dst_ortho_yz.ActualHeight * rel_mag;
            dst_z *= dst_ortho_zx.ActualHeight * rel_mag;

            xmax = dst_ortho_xy.ActualWidth * rel_mag - dst_sv_xy.ViewportWidth;
            ymax = dst_ortho_yz.ActualHeight * rel_mag - dst_sv_yz.ViewportHeight;
            zmax = dst_ortho_zx.ActualHeight * rel_mag - dst_sv_zx.ViewportHeight;

            dst_x -= dst_sv_xy.ViewportWidth / 2;
            if (dst_x < 0.0) dst_x = 0.0;
            if (dst_x > xmax) dst_x = xmax;

            dst_y -= dst_sv_yz.ViewportHeight / 2;
            if (dst_y < 0.0) dst_y = 0.0;
            if (dst_y > ymax) dst_y = ymax;

            dst_z -= dst_sv_zx.ViewportHeight / 2;
            if (dst_z < 0.0) dst_z = 0.0;
            if (dst_z > zmax) dst_z = zmax;

            viewSizeChange(view_src);
            viewSizeChange(view_dst);

            src_sv_xy.ScrollToHorizontalOffset(src_x);
            src_sv_xy.ScrollToVerticalOffset(src_y);
            src_sv_yz.ScrollToHorizontalOffset(src_z);
            src_sv_yz.ScrollToVerticalOffset(src_y);
            src_sv_zx.ScrollToHorizontalOffset(src_x);
            src_sv_zx.ScrollToVerticalOffset(src_z);

            dst_sv_xy.ScrollToHorizontalOffset(dst_x);
            dst_sv_xy.ScrollToVerticalOffset(dst_y);
            dst_sv_yz.ScrollToHorizontalOffset(dst_z);
            dst_sv_yz.ScrollToVerticalOffset(dst_y);
            dst_sv_zx.ScrollToHorizontalOffset(dst_x);
            dst_sv_zx.ScrollToVerticalOffset(dst_z);
        }

        private void ScrollViewer_ScrollChanged(object sender, ScrollChangedEventArgs e)
        {
            if (imageW <= 0 || imageH <= 0 || imageZ <= 0) return;
            /*
            var parent = LogicalTreeHelper.GetParent((DependencyObject)sender);
            if (parent is Grid)
            {
                Grid view = (Grid)parent;
                debug_cmp_txt.Text = view.Name;

                var svs = view.Children.Cast<UIElement>().Where(i => Grid.GetRow(i) == 0 && Grid.GetColumn(i) == 0 && i is ScrollViewer);
                if (svs.Count() == 0) return;
                ScrollViewer sv_xy = (ScrollViewer)svs.ElementAt(0);
               
                svs = view.Children.Cast<UIElement>().Where(i => Grid.GetRow(i) == 2 && Grid.GetColumn(i) == 0 && i is ScrollViewer);
                if (svs.Count() == 0) return;
                ScrollViewer sv_zx = (ScrollViewer)svs.ElementAt(0);
                var cont = sv_zx.Content;
                if (cont == null) return;
                if (!(cont is Grid)) return;
                Grid zxpanel = (Grid)cont;
                
                svs = view.Children.Cast<UIElement>().Where(i => Grid.GetRow(i) == 0 && Grid.GetColumn(i) == 2 && i is ScrollViewer);
                if (svs.Count() == 0) return;
                ScrollViewer sv_yz = (ScrollViewer)svs.ElementAt(0);
                cont = sv_yz.Content;
                if (cont == null) return;
                if (!(cont is Grid)) return;
                Grid yzpanel = (Grid)cont;

                if ((ScrollViewer)sender == sv_zx)
                {
                    sv_yz.ScrollToHorizontalOffset(sv_zx.VerticalOffset);
                }
                else if ((ScrollViewer)sender == sv_yz)
                {
                    sv_zx.ScrollToVerticalOffset(sv_yz.HorizontalOffset);
                }

                sv_zx.ScrollToHorizontalOffset(sv_xy.HorizontalOffset);
                sv_yz.ScrollToVerticalOffset(sv_xy.VerticalOffset);
            }
            */
            if (!((ScrollViewer)sender).IsMouseOver) return;

            if ((ScrollViewer)sender == src_sv_zx)
            {
                src_sv_yz.ScrollToHorizontalOffset(src_sv_zx.VerticalOffset);
                
                dst_sv_zx.ScrollToVerticalOffset(src_sv_zx.VerticalOffset * (dst_sv_zx.ViewportHeight / src_sv_zx.ViewportHeight));
                dst_sv_yz.ScrollToHorizontalOffset(src_sv_zx.VerticalOffset * (dst_sv_zx.ViewportHeight / src_sv_zx.ViewportHeight));
            }
            else if ((ScrollViewer)sender == src_sv_yz)
            {
                src_sv_zx.ScrollToVerticalOffset(src_sv_yz.HorizontalOffset);
                
                dst_sv_yz.ScrollToHorizontalOffset(src_sv_yz.HorizontalOffset * (dst_sv_yz.ViewportWidth / src_sv_yz.ViewportWidth));
                dst_sv_zx.ScrollToVerticalOffset(src_sv_yz.HorizontalOffset * (dst_sv_yz.ViewportWidth / src_sv_yz.ViewportWidth));
            }
            else if ((ScrollViewer)sender == src_sv_xy)
            {
                src_sv_zx.ScrollToHorizontalOffset(src_sv_xy.HorizontalOffset);
                src_sv_yz.ScrollToVerticalOffset(src_sv_xy.VerticalOffset);
                dst_sv_zx.ScrollToHorizontalOffset(src_sv_xy.HorizontalOffset * (dst_sv_xy.ViewportWidth / src_sv_xy.ViewportWidth));
                dst_sv_yz.ScrollToVerticalOffset(src_sv_xy.VerticalOffset * (dst_sv_xy.ViewportHeight / src_sv_xy.ViewportHeight));

                dst_sv_xy.ScrollToHorizontalOffset(src_sv_xy.HorizontalOffset * (dst_sv_xy.ViewportWidth / src_sv_xy.ViewportWidth));
                dst_sv_xy.ScrollToVerticalOffset(src_sv_xy.VerticalOffset * (dst_sv_xy.ViewportHeight / src_sv_xy.ViewportHeight));
            }
            else if ((ScrollViewer)sender == dst_sv_zx)
            {
                dst_sv_yz.ScrollToHorizontalOffset(dst_sv_zx.VerticalOffset);
                
                src_sv_zx.ScrollToVerticalOffset(dst_sv_zx.VerticalOffset * (src_sv_zx.ViewportHeight / dst_sv_zx.ViewportHeight));
                src_sv_yz.ScrollToHorizontalOffset(dst_sv_zx.VerticalOffset * (src_sv_zx.ViewportHeight / dst_sv_zx.ViewportHeight));
            }
            else if ((ScrollViewer)sender == dst_sv_yz)
            {
                dst_sv_zx.ScrollToVerticalOffset(dst_sv_yz.HorizontalOffset);
                
                src_sv_yz.ScrollToHorizontalOffset(dst_sv_yz.HorizontalOffset * (src_sv_yz.ViewportWidth / dst_sv_yz.ViewportWidth));
                src_sv_zx.ScrollToVerticalOffset(dst_sv_yz.HorizontalOffset * (src_sv_yz.ViewportWidth / dst_sv_yz.ViewportWidth));
            }
            else if ((ScrollViewer)sender == dst_sv_xy)
            {
                dst_sv_zx.ScrollToHorizontalOffset(dst_sv_xy.HorizontalOffset);
                dst_sv_yz.ScrollToVerticalOffset(dst_sv_xy.VerticalOffset);
                src_sv_zx.ScrollToHorizontalOffset(dst_sv_xy.HorizontalOffset * (src_sv_xy.ViewportWidth / dst_sv_xy.ViewportWidth));
                src_sv_yz.ScrollToVerticalOffset(dst_sv_xy.VerticalOffset * (src_sv_xy.ViewportHeight / dst_sv_xy.ViewportHeight));

                src_sv_xy.ScrollToHorizontalOffset(dst_sv_xy.HorizontalOffset * (src_sv_xy.ViewportWidth / dst_sv_xy.ViewportWidth));
                src_sv_xy.ScrollToVerticalOffset(dst_sv_xy.VerticalOffset * (src_sv_xy.ViewportHeight / dst_sv_xy.ViewportHeight));
            }
           
        }


    }
}
