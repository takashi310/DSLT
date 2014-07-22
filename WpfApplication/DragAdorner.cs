using System;
using System.Runtime.InteropServices;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media;
using System.Windows.Documents;
using System.Windows.Shapes;
using System.Windows.Interop;

namespace WpfApplication1
{
    class DragAdorner : Adorner
    {
        protected UIElement _child;
        protected double XCenter;
        protected double YCenter;

        public DragAdorner(UIElement owner) : base(owner) { }

        public DragAdorner(UIElement owner, UIElement adornElement, double opacity, Point dragPos)
            : base(owner)
        {
            var _brush = new VisualBrush(adornElement) { Opacity = opacity };
            var b = VisualTreeHelper.GetDescendantBounds(adornElement);
            var r = new Rectangle() { Width = b.Width, Height = b.Height };

            XCenter = dragPos.X;// r.Width / 2;
            YCenter = dragPos.Y;// r.Height / 2;

            r.Fill = _brush;
            _child = r;
        }


        private double _leftOffset;
        public double LeftOffset
        {
            get { return _leftOffset; }
            set
            {
                _leftOffset = value - XCenter;
                UpdatePosition();
            }
        }

        private double _topOffset;
        public double TopOffset
        {
            get { return _topOffset; }
            set
            {
                _topOffset = value - YCenter;
                UpdatePosition();
            }
        }

        private void UpdatePosition()
        {
            var adorner = this.Parent as AdornerLayer;
            if (adorner != null)
            {
                adorner.Update(this.AdornedElement);
            }
        }

        protected override Visual GetVisualChild(int index)
        {
            return _child;
        }

        protected override int VisualChildrenCount
        {
            get { return 1; }
        }

        protected override Size MeasureOverride(Size finalSize)
        {
            _child.Measure(finalSize);
            return _child.DesiredSize;
        }
        protected override Size ArrangeOverride(Size finalSize)
        {

            _child.Arrange(new Rect(_child.DesiredSize));
            return finalSize;
        }

        public override GeneralTransform GetDesiredTransform(GeneralTransform transform)
        {
            var result = new GeneralTransformGroup();
            result.Children.Add(base.GetDesiredTransform(transform));
            result.Children.Add(new TranslateTransform(_leftOffset, _topOffset));
            return result;
        }
    }

    public static class CursorInfo
    {
        [DllImport("user32.dll")]
        private static extern void GetCursorPos(out POINT pt);

        [DllImport("user32.dll")]
        private static extern int ScreenToClient(IntPtr hwnd, ref POINT pt);

        private struct POINT
        {
            public UInt32 X;
            public UInt32 Y;
        }

        ///<summary>
        ///カーソルのScreen座標を返します
        ///</summary>
        public static Point GetNowPosition(Visual v)
        {
            POINT p;
            GetCursorPos(out p);

            var source = HwndSource.FromVisual(v) as HwndSource;
            var hwnd = source.Handle;

            //ScreenToClient(hwnd, ref p);
            return new Point(p.X, p.Y);
        }
    }
}
