import threading
from tkinter import *
from time import sleep


class App(Tk):
    NO_VALUE_SELECTED = 'NO_VALUE_SELECTED'

    def __init__(self):
        super().__init__()
        self.bg = '#afeeee'
        self.title("Forecasting the GDP of states")
        self.configure(background=self.bg)

        # Border Section
        self.border = Label(self, text='  ', background=self.bg)
        self.border.grid(column=0, row=0, rowspan=10)

        self.border = Label(self, text='  ', background=self.bg)
        self.border.grid(column=3, row=0, rowspan=10)

        # Select model type section
        self.select_model_type_label = Label(self, text='Select model type:', background=self.bg)
        self.select_model_type_label.grid(column=1, row=0, columnspan=2)

        self.model_type = StringVar()
        self.model_type.set(self.NO_VALUE_SELECTED)

        self.classic_models_radio_btn = Radiobutton(self, text='Classic models', variable=self.model_type,
                                                               background=self.bg,
                                                               command=self.select_classic_models,
                                                               value='classic')
        self.classic_models_radio_btn.grid(column=1, row=1, sticky='w')

        self.neural_models_radio_btn = Radiobutton(self, text='Neural models', variable=self.model_type,
                                                   background=self.bg, command=self.select_neural_models,
                                                   value='neural')
        self.neural_models_radio_btn.grid(column=2, row=1, sticky='w')

        self.select_model_label = Label(self, text='Select classic model:', background=self.bg)




        """
        self.make_predict_button = Button(self, text="make predict", width=20,
                                          command=self.make_predict)
        self.make_predict_button.grid(column=1, row=10, sticky='w')
        """

        # Classic models section
        self.classic_models_fields = []

        self.classic_model_type = StringVar()
        self.classic_model_type.set(self.NO_VALUE_SELECTED)
        self.select_classic_model_label = Label(self, text='Classic models:', background=self.bg)

        self.linear_regression_radio_btn = Radiobutton(self, text='Linear regression', variable=self.classic_model_type,
                                                       background=self.bg, value='linear_regression',
                                                       command=self.select_classic_regression_model)

        self.svm_radio_btn = Radiobutton(self, text='SVM', variable=self.classic_model_type,
                                         background=self.bg, value='svm',
                                         command=self.select_classic_regression_model)

        self.tree_radio_btn = Radiobutton(self, text='Decision Tree', variable=self.classic_model_type,
                                          background=self.bg, value='decision_tree',
                                          command=self.select_classic_regression_model)

        # Classic ensemble models section

        self.ensemble_models_type = StringVar()
        self.ensemble_models_type.set(self.NO_VALUE_SELECTED)

        self.random_forest_radio_btn = Radiobutton(self, text='Random forest', variable=self.ensemble_models_type,
                                                   background=self.bg, value='random_forest',
                                                   command=self.select_ensemble_model_without_choice)

        self.select_ensemble_model_label = Label(self, text='Ensemble models:', background=self.bg)

        self.bagging_radio_btn = Radiobutton(self, text='Bagging', variable=self.ensemble_models_type,
                                             background=self.bg, value='bagging')

        self.grad_boosting_btn = Radiobutton(self, text='Gradient boosting', variable=self.ensemble_models_type,
                                             background=self.bg, value='grad_boosting')

        self.ada_boosting_btn = Radiobutton(self, text='AdaBoosting', variable=self.ensemble_models_type,
                                            background=self.bg, value='ada_boosting')

        # Neural models section
        self.neural_models_fields = []

        self.neural_models_type = StringVar()
        self.neural_models_type.set(self.NO_VALUE_SELECTED)

        self.select_neural_model_label = Label(self, text='Select neural model:', background=self.bg)
        # Feedforward neural network
        self.feed_froward_model_radio_btn = Radiobutton(self, text='Feedforward neural network',
                                                        variable=self.neural_models_type,
                                                        background=self.bg, value='feed_forward')

        self.gru_dense_model_radio_btn = Radiobutton(self, text='GRU + Dense neural network',
                                                     variable=self.neural_models_type,
                                                     background=self.bg, value='gru_dense')

        self.lstm_dense_model_radio_btn = Radiobutton(self, text='LSTM + Dense neural network',
                                                      variable=self.neural_models_type,
                                                      background=self.bg, value='lstm_dense')

        self.conv_gru_dense_model_radio_btn = Radiobutton(self, text='Conv + GRU + Dense neural network',
                                                          variable=self.neural_models_type,
                                                          background=self.bg, value='conv_gru_dense')

        self.conv_lstm_dense_model_radio_btn = Radiobutton(self, text='Conv + GRU + Dense neural network',
                                                           variable=self.neural_models_type,
                                                           background=self.bg, value='conv_lstm_dense')

    def select_ensemble_model_without_choice(self):
        self.classic_model_type.set(self.NO_VALUE_SELECTED)

    def select_classic_regression_model(self):
        # TODO
        print(self.ensemble_models_type)

    def select_neural_models(self):
        for field in self.classic_models_fields:
            field.destroy()

        self.select_neural_model_label = Label(self, text='Select neural model:', background=self.bg)
        self.select_neural_model_label.grid(column=1, row=2, columnspan=2)

        self.feed_froward_model_radio_btn = Radiobutton(self, text='Feedforward neural network',
                                                        variable=self.neural_models_type,
                                                        background=self.bg, value='feed_forward')
        self.feed_froward_model_radio_btn.grid(column=1, row=3, sticky='w')

        self.gru_dense_model_radio_btn = Radiobutton(self, text='GRU + Dense neural network',
                                                     variable=self.neural_models_type,
                                                     background=self.bg, value='gru_dense')
        self.gru_dense_model_radio_btn.grid(column=1, row=4, sticky='w')

        self.lstm_dense_model_radio_btn = Radiobutton(self, text='LSTM + Dense neural network',
                                                      variable=self.neural_models_type,
                                                      background=self.bg, value='lstm_dense')
        self.lstm_dense_model_radio_btn.grid(column=1, row=5, sticky='w')

        self.conv_gru_dense_model_radio_btn = Radiobutton(self, text='Conv + GRU + Dense neural network',
                                                          variable=self.neural_models_type,
                                                          background=self.bg, value='conv_gru_dense')
        self.conv_gru_dense_model_radio_btn.grid(column=1, row=6, sticky='w')

        self.conv_lstm_dense_model_radio_btn = Radiobutton(self, text='Conv + GRU + Dense neural network',
                                                           variable=self.neural_models_type,
                                                           background=self.bg, value='conv_lstm_dense')
        self.conv_lstm_dense_model_radio_btn.grid(column=1, row=7, sticky='w')

        self.neural_models_fields = [
            self.select_neural_model_label,
            self.feed_froward_model_radio_btn,
            self.gru_dense_model_radio_btn,
            self.lstm_dense_model_radio_btn,
            self.conv_gru_dense_model_radio_btn,
            self.conv_lstm_dense_model_radio_btn
        ]

    # def make_predict(self):
    #    print(self.models_state.get())

    def select_classic_models(self):
        for field in self.neural_models_fields :
            field.destroy()

        self.select_model_label = Label(self, text='Select model:', background=self.bg)
        self.select_model_label.grid(column=1, row=2, columnspan=2)

        # Classic models section

        self.select_classic_model_label = Label(self, text='Select classic model:', background=self.bg)
        self.select_classic_model_label.grid(column=1, row=3)

        self.linear_regression_radio_btn = Radiobutton(self, text='Linear regression', variable=self.classic_model_type,
                                                       background=self.bg, value='linear_regression',
                                                       command=self.select_classic_regression_model)
        self.linear_regression_radio_btn.grid(column=1, row=4, sticky='w')

        self.svm_radio_btn = Radiobutton(self, text='SVM', variable=self.classic_model_type,
                                         background=self.bg, value='svm',
                                         command=self.select_classic_regression_model)
        self.svm_radio_btn.grid(column=1, row=5, sticky='w')

        self.tree_radio_btn = Radiobutton(self, text='Decision Tree', variable=self.classic_model_type,
                                          background=self.bg, value='decision_tree',
                                          command=self.select_classic_regression_model)

        self.tree_radio_btn.grid(column=1, row=6, sticky='w')

        self.select_ensemble_model_label = Label(self, text='Ensemble models:', background=self.bg)
        self.select_ensemble_model_label.grid(column=2, row=3)

        # Ensemble models that do not have a choice of base model section

        self.random_forest_radio_btn = Radiobutton(self, text='Random forest', variable=self.ensemble_models_type,
                                                   background=self.bg, value='random_forest',
                                                   command=self.select_ensemble_model_without_choice)
        self.random_forest_radio_btn.grid(column=2, row=4, sticky='w')

        # Classic ensemble models section

        self.bagging_radio_btn = Radiobutton(self, text='Bagging', variable=self.ensemble_models_type,
                                             background=self.bg, value='bagging')
        self.bagging_radio_btn.grid(column=2, row=5, sticky='w')

        self.grad_boosting_btn = Radiobutton(self, text='Gradient boosting', variable=self.ensemble_models_type,
                                             background=self.bg, value='grad_boosting')
        self.grad_boosting_btn.grid(column=2, row=6, sticky='w')

        self.ada_boosting_btn = Radiobutton(self, text='AdaBoosting', variable=self.ensemble_models_type,
                                            background=self.bg, value='ada_boosting')
        self.ada_boosting_btn.grid(column=2, row=7, sticky='w')

        self.classic_models_fields = [
            self.select_model_label,
            self.select_classic_model_label,
            self.linear_regression_radio_btn,
            self.svm_radio_btn,
            self.tree_radio_btn,
            self.random_forest_radio_btn,
            self.select_ensemble_model_label,
            self.bagging_radio_btn,
            self.grad_boosting_btn,
            self.ada_boosting_btn
        ]




if __name__ == '__main__':
    window = App()
    window.mainloop()
