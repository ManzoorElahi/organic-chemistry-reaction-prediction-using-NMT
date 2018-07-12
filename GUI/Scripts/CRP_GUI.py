# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'CRP_GUI2.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from keras.layers import Bidirectional, Concatenate, Dot, Input, LSTM
from keras.layers import RepeatVector, Dense, Activation
from keras.models import Model
import numpy as np
from nmt_utils import *
from rdkit import Chem
from rdkit.Chem import Draw


Tx = 17
Ty = 19

reactants_vocab={'#': 0,
                 '(': 1,
                 ')': 2,
                 '-': 3,
                 '.': 4,
                 '/': 5,
                 '1': 6,
                 '2': 7,
                 '3': 8,
                 '<pad>': 86,
                 '<unk>': 85,
                 '=': 9,
                 'B': 10,
                 'Br': 11,
                 'C': 12,
                 'Cl': 13,
                 'F': 14,
                 'I': 15,
                 'N': 16,
                 'O': 17,
                 'P': 18,
                 'S': 19,
                 '[Al+]': 20,
                 '[As]': 21,
                 '[BH3-]': 22,
                 '[Br-]': 23,
                 '[C-]': 24,
                 '[C@@H]': 25,
                 '[C@@]': 26,
                 '[C@H]': 27,
                 '[C@]': 28,
                 '[Cl-]': 29,
                 '[Cu]': 30,
                 '[F-]': 31,
                 '[Ga+2]': 32,
                 '[Ga+]': 33,
                 '[GeH]': 34,
                 '[Hg]': 35,
                 '[I-]': 36,
                 '[IH2+]': 37,
                 '[K]': 38,
                 '[Li]': 39,
                 '[Mg+2]': 40,
                 '[Mg+]': 41,
                 '[Mg]': 42,
                 '[N+]': 43,
                 '[N-]': 44,
                 '[NH+]': 45,
                 '[NH-]': 46,
                 '[NH2+]': 47,
                 '[NH2-]': 48,
                 '[NH3+]': 49,
                 '[NH4+]': 50,
                 '[Na+]': 51,
                 '[Na]': 52,
                 '[O+]': 53,
                 '[O-2]': 54,
                 '[O-]': 55,
                 '[OH-]': 56,
                 '[PH2]': 57,
                 '[PH3+]': 58,
                 '[PH]': 59,
                 '[Pb]': 60,
                 '[S+]': 61,
                 '[S-2]': 62,
                 '[S-]': 63,
                 '[SH-]': 64,
                 '[Se]': 65,
                 '[SiH2]': 66,
                 '[SiH3]': 67,
                 '[SiH]': 68,
                 '[Si]': 69,
                 '[Sn]': 70,
                 '[Zn+2]': 71,
                 '[Zn]': 72,
                 '[cH+]': 73,
                 '[cH-]': 74,
                 '[n+]': 75,
                 '[n-]': 76,
                 '[nH+]': 77,
                 '[nH]': 78,
                 '[se]': 79,
                 '\\': 80,
                 'c': 81,
                 'n': 82,
                 'o': 83,
                 's': 84}


products_vocab ={'#': 0,
                 '(': 1,
                 ')': 2,
                 '-': 3,
                 '/': 4,
                 '1': 5,
                 '2': 6,
                 '3': 7,
                 '<pad>': 55,
                 '<unk>': 54,
                 '=': 8,
                 'B': 9,
                 'Br': 10,
                 'C': 11,
                 'Cl': 12,
                 'F': 13,
                 'I': 14,
                 'N': 15,
                 'O': 16,
                 'P': 17,
                 'S': 18,
                 '[AlH]': 19,
                 '[As]': 20,
                 '[C-]': 21,
                 '[C@@H]': 22,
                 '[C@@]': 23,
                 '[C@H]': 24,
                 '[C@]': 25,
                 '[Ga]': 26,
                 '[Ge]': 27,
                 '[Hg]': 28,
                 '[Li]': 29,
                 '[Mg]': 30,
                 '[N+]': 31,
                 '[N-]': 32,
                 '[Na]': 33,
                 '[O-]': 34,
                 '[PH]': 35,
                 '[S@@]': 36,
                 '[SH]': 37,
                 '[Se]': 38,
                 '[SiH2]': 39,
                 '[SiH3]': 40,
                 '[SiH]': 41,
                 '[Si]': 42,
                 '[SnH]': 43,
                 '[Sn]': 44,
                 '[Zn]': 45,
                 '[n+]': 46,
                 '[nH]': 47,
                 '[se]': 48,
                 '\\': 49,
                 'c': 50,
                 'n': 51,
                 'o': 52,
                 's': 53}
    

inv_products_vocab = {v:k for k,v in products_vocab.items()}

   
repeator = RepeatVector(Tx)
concatenator = Concatenate(axis=-1)
densor = Dense(1, activation = "relu")
activator = Activation(softmax, name='attention_weights')
dotor = Dot(axes = 1)


def one_step_attention(a, s_prev):
    """
    Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights
    "alphas" and the hidden states "a" of the Bi-LSTM.
    
    Arguments:
    a -- hidden state output of the Bi-LSTM, numpy-array of shape (m, Tx, 2*n_a)
    s_prev -- previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s)
    
    Returns:
    context -- context vector, input of the next (post-attetion) LSTM cell
    """
    
    s_prev = repeator(s_prev)
    concat = concatenator([a,s_prev])
    e = densor(concat)
    alphas = activator(e)
    context = dotor([alphas,a])
    
    return context


n_a = 256
n_s = 512
post_activation_LSTM_cell = LSTM(n_s, return_state = True)
output_layer = Dense(len(products_vocab), activation=softmax)


def model(Tx, Ty, n_a, n_s, reactants_vocab_size, products_vocab_size):
    """
    Arguments:
    Tx -- length of the input sequence
    Ty -- length of the output sequence
    n_a -- hidden state size of the Bi-LSTM
    n_s -- hidden state size of the post-attention LSTM
    human_vocab_size -- size of the python dictionary "human_vocab"
    machine_vocab_size -- size of the python dictionary "machine_vocab"

    Returns:
    model -- Keras model instance
    """
    
    X = Input(shape=(Tx, reactants_vocab_size))
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')
    s = s0
    c = c0    
    outputs = []   
    a = Bidirectional(LSTM(n_a, return_sequences=True))(X)    
    for t in range(Ty):   
        context = one_step_attention(a, s)       
        s, _, c = post_activation_LSTM_cell(inputs = context, initial_state = [s,c])
        out = output_layer(s)        
        outputs.append(out)
    model = Model(inputs = [X, s0, c0], outputs = outputs)
    return model


model = model(Tx, Ty, n_a, n_s, len(reactants_vocab), len(products_vocab))

s0 = np.zeros((1, n_s))
c0 = np.zeros((1, n_s))


model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics = ['accuracy'])
model.load_weights('weights256.h5')


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(387, 505)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_7.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label, 0, QtCore.Qt.AlignBottom)
        self.verticalLayout_7.addLayout(self.verticalLayout)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.LEInSMILE = QtWidgets.QLineEdit(self.centralwidget)
        self.LEInSMILE.setObjectName("LEInSMILE")
        self.verticalLayout_3.addWidget(self.LEInSMILE)
        self.verticalLayout_7.addLayout(self.verticalLayout_3)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.BClear = QtWidgets.QPushButton(self.centralwidget)
        self.BClear.setObjectName("BClear")
        self.horizontalLayout.addWidget(self.BClear)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.BPredict = QtWidgets.QPushButton(self.centralwidget)
        self.BPredict.setObjectName("BPredict")
        self.horizontalLayout.addWidget(self.BPredict)
        self.verticalLayout_7.addLayout(self.horizontalLayout)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_4.addWidget(self.label_2, 0, QtCore.Qt.AlignBottom)
        self.verticalLayout_7.addLayout(self.verticalLayout_4)
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.tBOutSMILE = QtWidgets.QTextBrowser(self.centralwidget)
        self.tBOutSMILE.setObjectName("tBOutSMILE")
        self.verticalLayout_5.addWidget(self.tBOutSMILE)
        self.verticalLayout_7.addLayout(self.verticalLayout_5)
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setObjectName("label_3")
        self.verticalLayout_6.addWidget(self.label_3, 0, QtCore.Qt.AlignBottom)
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setMaximumSize(QtCore.QSize(617, 400))
        self.label_4.setBaseSize(QtCore.QSize(300, 300))
        self.label_4.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.label_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_4.setText("")
        self.label_4.setObjectName("label_4")
        self.verticalLayout_6.addWidget(self.label_4)
        self.verticalLayout_6.setStretch(0, 1)
        self.verticalLayout_6.setStretch(1, 10)
        self.verticalLayout_7.addLayout(self.verticalLayout_6)
        self.verticalLayout_7.setStretch(0, 1)
        self.verticalLayout_7.setStretch(1, 1)
        self.verticalLayout_7.setStretch(2, 1)
        self.verticalLayout_7.setStretch(3, 1)
        self.verticalLayout_7.setStretch(4, 1)
        self.verticalLayout_7.setStretch(5, 10)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 387, 20))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.BClear.clicked.connect(self.clearSMILE)
        self.BPredict.clicked.connect(self.predictSMILE)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


    def clearSMILE(self):

        self.LEInSMILE.clear()


    def showSMILE(self, smile):

        self.tBOutSMILE.append(smile)


    def showMolecule(self, smile):

        m = Chem.MolFromSmiles(smile)
        Draw.MolToFile(m, fileName = 'im.png', size=(300, 300))
        pixmap = QtGui.QPixmap('im.png')
        #scaledpixmap = pixmap.scaled(self.label.size())
        self.label_4.setPixmap(pixmap)


    def clearOutput(self):

        self.tBOutSMILE.clear()


    def getSMILE(self):

        seq = self.LEInSMILE.text()

        return seq


    def predictSMILE(self):

        self.clearOutput()
        smile = self.getSMILE()
        X = smile
        X = X.split()
        X = np.array(string_to_int(X, Tx, reactants_vocab))
        Xoh = np.array(list(map(lambda x: to_categorical(x, num_classes=len(reactants_vocab)), X)))
        Xoh = np.expand_dims(Xoh, axis=0)
        
        prediction = model.predict([Xoh, s0, c0])
        
        pad = '<pad>'
        prediction = np.argmax(np.array(prediction)[:,0,:], axis = 1)
        prediction = int_to_string(prediction,inv_products_vocab)
        
        pred = []
        for x in prediction:
            if x != pad:
                pred.append(x)
        pred = ''.join(pred)
        
        self.showSMILE(pred)
        self.showMolecule(pred)
        
        
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "CRP"))
        self.label.setText(_translate("MainWindow", "Enter SMILE:"))
        self.BClear.setText(_translate("MainWindow", "Clear"))
        self.BPredict.setText(_translate("MainWindow", "Predict"))
        self.label_2.setText(_translate("MainWindow", "Predicted SMILE:"))
        self.label_3.setText(_translate("MainWindow", "Molecule:"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

