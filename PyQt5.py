# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from PyQt5.QtWidgets import QApplication, QWidget
a = QApplication([])
w = QWidget()
w.show()

# <codecell>

from PyQt4 import QtGui

# <codecell>

app = QtGui.QApplication(sys.argv)

w = QtGui.QWidget()
w.resize(250, 150)
w.move(300, 300)
w.setWindowTitle('Simple')
w.show()

