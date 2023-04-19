# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 11:24:41 2020

@author: cglos
"""

from PyQt5.QtWidgets import (
    QWidget, QApplication, QProgressBar, QMainWindow,
    QHBoxLayout, QPushButton
)

from PyQt5.QtCore import (
    Qt, QObject, pyqtSignal, pyqtSlot, QRunnable, QThreadPool
)
import time


class WorkerSignals(QObject):
    progress = pyqtSignal(int)


class JobRunner(QRunnable):
    
    signals = WorkerSignals()
    
    def __init__(self):
        super().__init__()
        
        self.is_paused = False
        self.is_killed = False
        
    @pyqtSlot()
    def run(self):
        for n in range(100):
            self.signals.progress.emit(n + 1)
            time.sleep(0.1)
            
            while self.is_paused:
                time.sleep(0)
                
            if self.is_killed:
                break
                
    def pause(self):
        self.is_paused = True
        
    def resume(self):
        self.is_paused = False
        
    def kill(self):
        self.is_killed = True


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        
        # Some buttons
        w = QWidget()
        l = QHBoxLayout()
        w.setLayout(l)
        
        btn_stop = QPushButton("Stop")
        btn_pause = QPushButton("Pause")
        btn_resume = QPushButton("Resume")
        
        l.addWidget(btn_stop)
        l.addWidget(btn_pause)
        l.addWidget(btn_resume)
        
        self.setCentralWidget(w)
       
        # Create a statusbar.
        self.status = self.statusBar()
        self.progress = QProgressBar()
        self.status.addPermanentWidget(self.progress)
        
        # Thread runner
        self.threadpool = QThreadPool()
        
        # Create a runner
        self.runner = JobRunner()
        self.runner.signals.progress.connect(self.update_progress)
        self.threadpool.start(self.runner)

        btn_stop.pressed.connect(self.runner.kill)
        btn_pause.pressed.connect(self.runner.pause)
        btn_resume.pressed.connect(self.runner.resume)
        
        self.show()
    
    def update_progress(self, n):
        self.progress.setValue(n)
        
app = QApplication([])
w = MainWindow()
app.exec_()