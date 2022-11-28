# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.12
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(486, 366)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMinimumSize(QtCore.QSize(0, 0))
        MainWindow.setMaximumSize(QtCore.QSize(6000, 3590))
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralWidget.sizePolicy().hasHeightForWidth())
        self.centralWidget.setSizePolicy(sizePolicy)
        self.centralWidget.setObjectName("centralWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralWidget)
        self.verticalLayout.setContentsMargins(11, 11, 11, 11)
        self.verticalLayout.setSpacing(6)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.horizontalLayout.setSpacing(6)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.widget = QtWidgets.QWidget(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget.sizePolicy().hasHeightForWidth())
        self.widget.setSizePolicy(sizePolicy)
        self.widget.setMinimumSize(QtCore.QSize(45, 256))
        self.widget.setMaximumSize(QtCore.QSize(45, 256))
        self.widget.setObjectName("widget")
        self.gridLayout = QtWidgets.QGridLayout(self.widget)
        self.gridLayout.setContentsMargins(5, 5, 5, 5)
        self.gridLayout.setSpacing(6)
        self.gridLayout.setObjectName("gridLayout")
        self.polylineButton = QtWidgets.QPushButton(self.widget)
        self.polylineButton.setMinimumSize(QtCore.QSize(35, 35))
        self.polylineButton.setMaximumSize(QtCore.QSize(35, 35))
        self.polylineButton.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("images/layer-shape-polyline.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.polylineButton.setIcon(icon)
        self.polylineButton.setCheckable(True)
        self.polylineButton.setObjectName("polylineButton")
        self.gridLayout.addWidget(self.polylineButton, 7, 0, 1, 1)
        self.polygonButton = QtWidgets.QPushButton(self.widget)
        self.polygonButton.setMinimumSize(QtCore.QSize(35, 35))
        self.polygonButton.setMaximumSize(QtCore.QSize(35, 35))
        self.polygonButton.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("images/layer-shape-polygon.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.polygonButton.setIcon(icon1)
        self.polygonButton.setCheckable(True)
        self.polygonButton.setObjectName("polygonButton")
        self.gridLayout.addWidget(self.polygonButton, 8, 0, 1, 1, QtCore.Qt.AlignTop)
        self.lineButton = QtWidgets.QPushButton(self.widget)
        self.lineButton.setMinimumSize(QtCore.QSize(35, 35))
        self.lineButton.setMaximumSize(QtCore.QSize(35, 35))
        self.lineButton.setText("")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("images/layer-shape-line.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.lineButton.setIcon(icon2)
        self.lineButton.setCheckable(True)
        self.lineButton.setObjectName("lineButton")
        self.gridLayout.addWidget(self.lineButton, 6, 0, 1, 1)
        self.penButton = QtWidgets.QPushButton(self.widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.penButton.sizePolicy().hasHeightForWidth())
        self.penButton.setSizePolicy(sizePolicy)
        self.penButton.setMinimumSize(QtCore.QSize(35, 35))
        self.penButton.setMaximumSize(QtCore.QSize(35, 35))
        self.penButton.setText("")
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap("images/pencil.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.penButton.setIcon(icon3)
        self.penButton.setCheckable(True)
        self.penButton.setObjectName("penButton")
        self.gridLayout.addWidget(self.penButton, 0, 0, 1, 1, QtCore.Qt.AlignLeft)
        self.eraserButton = QtWidgets.QPushButton(self.widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.eraserButton.sizePolicy().hasHeightForWidth())
        self.eraserButton.setSizePolicy(sizePolicy)
        self.eraserButton.setMinimumSize(QtCore.QSize(35, 35))
        self.eraserButton.setMaximumSize(QtCore.QSize(35, 35))
        self.eraserButton.setText("")
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap("images/eraser.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.eraserButton.setIcon(icon4)
        self.eraserButton.setCheckable(True)
        self.eraserButton.setObjectName("eraserButton")
        self.gridLayout.addWidget(self.eraserButton, 1, 0, 1, 1)
        self.clearButton = QtWidgets.QPushButton(self.widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.clearButton.sizePolicy().hasHeightForWidth())
        self.clearButton.setSizePolicy(sizePolicy)
        self.clearButton.setMinimumSize(QtCore.QSize(35, 35))
        self.clearButton.setMaximumSize(QtCore.QSize(35, 35))
        self.clearButton.setText("")
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap("images/broom.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.clearButton.setIcon(icon5)
        self.clearButton.setObjectName("clearButton")
        self.gridLayout.addWidget(self.clearButton, 5, 0, 1, 1)
        self.horizontalLayout.addWidget(self.widget, 0, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignTop)
        self.verticalLayout.addLayout(self.horizontalLayout)
        MainWindow.setCentralWidget(self.centralWidget)
        self.menuBar = QtWidgets.QMenuBar(MainWindow)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 486, 23))
        self.menuBar.setObjectName("menuBar")
        self.menuFIle = QtWidgets.QMenu(self.menuBar)
        self.menuFIle.setObjectName("menuFIle")
        self.menuEdit = QtWidgets.QMenu(self.menuBar)
        self.menuEdit.setObjectName("menuEdit")
        self.menuImage = QtWidgets.QMenu(self.menuBar)
        self.menuImage.setObjectName("menuImage")
        self.menuHelp = QtWidgets.QMenu(self.menuBar)
        self.menuHelp.setObjectName("menuHelp")
        MainWindow.setMenuBar(self.menuBar)
        self.statusBar = QtWidgets.QStatusBar(MainWindow)
        self.statusBar.setObjectName("statusBar")
        MainWindow.setStatusBar(self.statusBar)
        self.fileToolbar = QtWidgets.QToolBar(MainWindow)
        self.fileToolbar.setAutoFillBackground(False)
        self.fileToolbar.setIconSize(QtCore.QSize(25, 25))
        self.fileToolbar.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.fileToolbar.setObjectName("fileToolbar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.fileToolbar)
        self.drawingToolbar = QtWidgets.QToolBar(MainWindow)
        self.drawingToolbar.setIconSize(QtCore.QSize(16, 16))
        self.drawingToolbar.setObjectName("drawingToolbar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.drawingToolbar)
        self.actionCopy = QtWidgets.QAction(MainWindow)
        self.actionCopy.setObjectName("actionCopy")
        self.actionClearImage = QtWidgets.QAction(MainWindow)
        self.actionClearImage.setObjectName("actionClearImage")
        self.actionOpenImage = QtWidgets.QAction(MainWindow)
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap("images/blue-folder-open-image.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionOpenImage.setIcon(icon6)
        self.actionOpenImage.setObjectName("actionOpenImage")
        self.actionSaveImage = QtWidgets.QAction(MainWindow)
        icon7 = QtGui.QIcon()
        icon7.addPixmap(QtGui.QPixmap("images/disk.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionSaveImage.setIcon(icon7)
        self.actionSaveImage.setObjectName("actionSaveImage")
        self.actionInvertColors = QtWidgets.QAction(MainWindow)
        self.actionInvertColors.setObjectName("actionInvertColors")
        self.actionFlipHorizontal = QtWidgets.QAction(MainWindow)
        self.actionFlipHorizontal.setObjectName("actionFlipHorizontal")
        self.actionFlipVertical = QtWidgets.QAction(MainWindow)
        self.actionFlipVertical.setObjectName("actionFlipVertical")
        self.actionNewImage = QtWidgets.QAction(MainWindow)
        icon8 = QtGui.QIcon()
        icon8.addPixmap(QtGui.QPixmap("images/document-image.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionNewImage.setIcon(icon8)
        self.actionNewImage.setObjectName("actionNewImage")
        self.actionBold = QtWidgets.QAction(MainWindow)
        self.actionBold.setCheckable(True)
        icon9 = QtGui.QIcon()
        icon9.addPixmap(QtGui.QPixmap("images/edit-bold.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionBold.setIcon(icon9)
        self.actionBold.setObjectName("actionBold")
        self.actionItalic = QtWidgets.QAction(MainWindow)
        self.actionItalic.setCheckable(True)
        icon10 = QtGui.QIcon()
        icon10.addPixmap(QtGui.QPixmap("images/edit-italic.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionItalic.setIcon(icon10)
        self.actionItalic.setObjectName("actionItalic")
        self.actionUnderline = QtWidgets.QAction(MainWindow)
        self.actionUnderline.setCheckable(True)
        icon11 = QtGui.QIcon()
        icon11.addPixmap(QtGui.QPixmap("images/edit-underline.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionUnderline.setIcon(icon11)
        self.actionUnderline.setObjectName("actionUnderline")
        self.actionFillShapes = QtWidgets.QAction(MainWindow)
        self.actionFillShapes.setCheckable(True)
        icon12 = QtGui.QIcon()
        icon12.addPixmap(QtGui.QPixmap("images/paint-can-color.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionFillShapes.setIcon(icon12)
        self.actionFillShapes.setObjectName("actionFillShapes")
        self.actionSaveGenerated = QtWidgets.QAction(MainWindow)
        self.actionSaveGenerated.setIcon(icon7)
        self.actionSaveGenerated.setObjectName("actionSaveGenerated")
        self.actionRandomSketch = QtWidgets.QAction(MainWindow)
        icon13 = QtGui.QIcon()
        icon13.addPixmap(QtGui.QPixmap("images/cake.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionRandomSketch.setIcon(icon13)
        self.actionRandomSketch.setObjectName("actionRandomSketch")
        self.menuFIle.addAction(self.actionNewImage)
        self.menuFIle.addAction(self.actionOpenImage)
        self.menuFIle.addAction(self.actionSaveImage)
        self.menuEdit.addAction(self.actionCopy)
        self.menuEdit.addSeparator()
        self.menuImage.addAction(self.actionInvertColors)
        self.menuImage.addSeparator()
        self.menuImage.addAction(self.actionFlipHorizontal)
        self.menuImage.addAction(self.actionFlipVertical)
        self.menuBar.addAction(self.menuFIle.menuAction())
        self.menuBar.addAction(self.menuEdit.menuAction())
        self.menuBar.addAction(self.menuImage.menuAction())
        self.menuBar.addAction(self.menuHelp.menuAction())
        self.fileToolbar.addAction(self.actionOpenImage)
        self.fileToolbar.addAction(self.actionRandomSketch)
        self.fileToolbar.addAction(self.actionSaveImage)
        self.fileToolbar.addAction(self.actionSaveGenerated)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Piecasso"))
        self.menuFIle.setTitle(_translate("MainWindow", "FIle"))
        self.menuEdit.setTitle(_translate("MainWindow", "Edit"))
        self.menuImage.setTitle(_translate("MainWindow", "Image"))
        self.menuHelp.setTitle(_translate("MainWindow", "Help"))
        self.fileToolbar.setWindowTitle(_translate("MainWindow", "toolBar"))
        self.drawingToolbar.setWindowTitle(_translate("MainWindow", "toolBar"))
        self.actionCopy.setText(_translate("MainWindow", "Copy"))
        self.actionCopy.setShortcut(_translate("MainWindow", "Ctrl+C"))
        self.actionClearImage.setText(_translate("MainWindow", "Clear Image"))
        self.actionOpenImage.setText(_translate("MainWindow", "Open"))
        self.actionSaveImage.setText(_translate("MainWindow", "Save Left"))
        self.actionInvertColors.setText(_translate("MainWindow", "Invert Colors"))
        self.actionFlipHorizontal.setText(_translate("MainWindow", "Flip Horizontal"))
        self.actionFlipVertical.setText(_translate("MainWindow", "Flip Vertical"))
        self.actionNewImage.setText(_translate("MainWindow", "New Image"))
        self.actionBold.setText(_translate("MainWindow", "Bold"))
        self.actionBold.setShortcut(_translate("MainWindow", "Ctrl+B"))
        self.actionItalic.setText(_translate("MainWindow", "Italic"))
        self.actionItalic.setShortcut(_translate("MainWindow", "Ctrl+I"))
        self.actionUnderline.setText(_translate("MainWindow", "Underline"))
        self.actionFillShapes.setText(_translate("MainWindow", "Fill Shapes?"))
        self.actionSaveGenerated.setText(_translate("MainWindow", "Save Right"))
        self.actionRandomSketch.setText(_translate("MainWindow", "Random"))


