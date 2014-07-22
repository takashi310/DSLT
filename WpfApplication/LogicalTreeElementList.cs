using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media;
using System.Windows.Documents;
using System.Windows.Shapes;
using System.Windows.Interop;
using System.Diagnostics;

namespace WpfApplication1
{
	class LogicalTreeElementList
	{
		public delegate void Enumerate(DependencyObject target, int level, Object option);

		private static void EnumerateVisualChildren(DependencyObject src, Enumerate func, int level, Object option)
		{
			//LogicalTreeHelper.GetChildrenはString型を返す場合があるのでチェック必須
			foreach (object obj in LogicalTreeHelper.GetChildren(src))
			{
				DependencyObject childElem = obj as DependencyObject;

				if (childElem != null)
				{
					func(childElem, level, option);

					EnumerateVisualChildren(childElem, func, level + 1, option);
				}
				
			}

		}

		public static void EnumerateVisualChildren(DependencyObject src, Enumerate func, Object option)
		{

			func(src, 0, option);

			EnumerateVisualChildren(src, func, 1, option);

		}

	}
}
