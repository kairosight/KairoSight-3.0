# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'KairoSight_WindowMDI.ui'
#
# Created by: PyQt5 UI code generator 5.14.2
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_WindowMDI(object):
    def setupUi(self, WindowMDI):
        WindowMDI.setObjectName("WindowMDI")
        WindowMDI.resize(1000, 700)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(WindowMDI.sizePolicy().hasHeightForWidth())
        WindowMDI.setSizePolicy(sizePolicy)
        self.centralwidget = QtWidgets.QWidget(WindowMDI)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.mdiArea = QtWidgets.QMdiArea(self.centralwidget)
        self.mdiArea.setObjectName("mdiArea")
        self.verticalLayout.addWidget(self.mdiArea)
        self.verticalLayout_2.addLayout(self.verticalLayout)
        WindowMDI.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(WindowMDI)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1000, 21))
        self.menubar.setObjectName("menubar")
        self.menuOpen = QtWidgets.QMenu(self.menubar)
        self.menuOpen.setObjectName("menuOpen")
        self.menuExport = QtWidgets.QMenu(self.menubar)
        self.menuExport.setObjectName("menuExport")
        WindowMDI.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(WindowMDI)
        self.statusbar.setObjectName("statusbar")
        WindowMDI.setStatusBar(self.statusbar)
        self.actionTIFF = QtWidgets.QAction(WindowMDI)
        self.actionTIFF.setObjectName("actionTIFF")
        self.actionClose = QtWidgets.QAction(WindowMDI)
        self.actionClose.setObjectName("actionClose")
        self.actionFolder = QtWidgets.QAction(WindowMDI)
        self.actionFolder.setObjectName("actionFolder")
        self.actionStart_ImagePrep = QtWidgets.QAction(WindowMDI)
        self.actionStart_ImagePrep.setObjectName("actionStart_ImagePrep")
        self.actionStart_Isolate = QtWidgets.QAction(WindowMDI)
        self.actionStart_Isolate.setObjectName("actionStart_Isolate")
        self.actionStart_Analyze = QtWidgets.QAction(WindowMDI)
        self.actionStart_Analyze.setObjectName("actionStart_Analyze")
        self.actionStart_Export = QtWidgets.QAction(WindowMDI)
        self.actionStart_Export.setObjectName("actionStart_Export")
        self.actionExport_CopyPaste = QtWidgets.QAction(WindowMDI)
        self.actionExport_CopyPaste.setObjectName("actionExport_CopyPaste")
        self.menuOpen.addAction(self.actionTIFF)
        self.menuOpen.addAction(self.actionFolder)
        self.menuOpen.addSeparator()
        self.menubar.addAction(self.menuOpen.menuAction())
        self.menubar.addAction(self.menuExport.menuAction())

        self.retranslateUi(WindowMDI)
        QtCore.QMetaObject.connectSlotsByName(WindowMDI)

    def retranslateUi(self, WindowMDI):
        _translate = QtCore.QCoreApplication.translate
        WindowMDI.setWindowTitle(_translate("WindowMDI", "KairoSight"))
        self.menuOpen.setTitle(_translate("WindowMDI", "Open"))
        self.menuExport.setTitle(_translate("WindowMDI", "Export"))
        self.actionTIFF.setText(_translate("WindowMDI", "TIFF"))
        self.actionTIFF.setToolTip(_translate("WindowMDI", ".tiff, .tif"))
        self.actionClose.setText(_translate("WindowMDI", "Close"))
        self.actionFolder.setText(_translate("WindowMDI", "Folder"))
        self.actionStart_ImagePrep.setText(_translate("WindowMDI", "Start wizard"))
        self.actionStart_Isolate.setText(_translate("WindowMDI", "Start wizard"))
        self.actionStart_Analyze.setText(_translate("WindowMDI", "Start wizard"))
        self.actionStart_Export.setText(_translate("WindowMDI", "Start wizard"))
        self.actionExport_CopyPaste.setText(_translate("WindowMDI", "for Copy + Paste"))
