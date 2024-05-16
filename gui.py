from tkinter import *


class App(Tk):
    NO_VALUE_SELECTED = 'NO_VALUE_SELECTED'

    def __init__(self):
        super().__init__()
        self.bg = '#afeeee'
        self.title("Forecasting the GDP of states")
        self.configure(background=self.bg)

        # Border Section
        self.border = Label(self, text='  ', background=self.bg)
        self.border.grid(column=0, row=0, rowspan=13)

        self.border = Label(self, text='  ', background=self.bg)
        self.border.grid(column=3, row=0, rowspan=13)

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

        # Classic models section
        self.classic_models_fields = []
        self.classic_model_type = StringVar()
        self.classic_model_type.set(self.NO_VALUE_SELECTED)

        # Classic ensemble models section
        self.ensemble_models_type = StringVar()
        self.ensemble_models_type.set(self.NO_VALUE_SELECTED)

        # Neural models section
        self.neural_models_fields = []
        self.neural_models_type = StringVar()
        self.neural_models_type.set(self.NO_VALUE_SELECTED)

        # enter_dataset_section
        self.enter_dataset_field_label = Label(self, text='Enter dataset path:', background=self.bg)
        self.enter_dataset_field = Text(width=30, height=1, bg='white', fg='black', wrap=WORD)
        self.empty_row = Label(self, text='  ', background=self.bg)
        self.predict_fields_was_drawn = False
        self.predict_counter = 1

    def select_ensemble_model_without_choice(self):
        self.classic_model_type.set(self.NO_VALUE_SELECTED)

    def select_classic_regression_model(self):
        if self.ensemble_models_type.get() != 'bagging':
            self.ensemble_models_type.set(self.NO_VALUE_SELECTED)

    def select_neural_models(self):
        for field in self.classic_models_fields:
            field.destroy()

        select_neural_model_label = Label(self, text='Select neural model:', background=self.bg)
        select_neural_model_label.grid(column=1, row=2, columnspan=2)

        feed_froward_model_radio_btn = Radiobutton(self, text='Feedforward neural network',
                                                   variable=self.neural_models_type,
                                                   background=self.bg, value='feed_forward')
        feed_froward_model_radio_btn.grid(column=1, row=3, sticky='w')

        gru_dense_model_radio_btn = Radiobutton(self, text='GRU + Dense neural network',
                                                variable=self.neural_models_type,
                                                background=self.bg, value='gru_dense')
        gru_dense_model_radio_btn.grid(column=1, row=4, sticky='w')

        lstm_dense_model_radio_btn = Radiobutton(self, text='LSTM + Dense neural network',
                                                 variable=self.neural_models_type,
                                                 background=self.bg, value='lstm_dense')
        lstm_dense_model_radio_btn.grid(column=1, row=5, sticky='w')

        conv_gru_dense_model_radio_btn = Radiobutton(self, text='Conv + GRU + Dense neural network',
                                                     variable=self.neural_models_type,
                                                     background=self.bg, value='conv_gru_dense')
        conv_gru_dense_model_radio_btn.grid(column=1, row=6, sticky='w')

        conv_lstm_dense_model_radio_btn = Radiobutton(self, text='Conv + GRU + Dense neural network',
                                                      variable=self.neural_models_type,
                                                      background=self.bg, value='conv_lstm_dense')
        conv_lstm_dense_model_radio_btn.grid(column=1, row=7, sticky='w')

        self.neural_models_fields = [
            select_neural_model_label,
            feed_froward_model_radio_btn,
            gru_dense_model_radio_btn,
            lstm_dense_model_radio_btn,
            conv_gru_dense_model_radio_btn,
            conv_lstm_dense_model_radio_btn
        ]
        self.drawing_dataset_and_predict_fields()

    def select_classic_models(self):
        for field in self.neural_models_fields:
            field.destroy()

        select_model_label = Label(self, text='Select model:', background=self.bg)
        select_model_label.grid(column=1, row=2, columnspan=2)

        # Classic models section

        select_classic_model_label = Label(self, text='Select classic model:', background=self.bg)
        select_classic_model_label.grid(column=1, row=3)

        linear_regression_radio_btn = Radiobutton(self, text='Linear regression', variable=self.classic_model_type,
                                                  background=self.bg, value='linear_regression',
                                                  command=self.select_classic_regression_model)
        linear_regression_radio_btn.grid(column=1, row=4, sticky='w')

        svm_radio_btn = Radiobutton(self, text='SVM', variable=self.classic_model_type,
                                    background=self.bg, value='svm',
                                    command=self.select_classic_regression_model)
        svm_radio_btn.grid(column=1, row=5, sticky='w')

        tree_radio_btn = Radiobutton(self, text='Decision Tree', variable=self.classic_model_type,
                                     background=self.bg, value='decision_tree',
                                     command=self.select_classic_regression_model)

        tree_radio_btn.grid(column=1, row=6, sticky='w')

        select_ensemble_model_label = Label(self, text='Ensemble models:', background=self.bg)
        select_ensemble_model_label.grid(column=2, row=3)

        # Ensemble models that do not have a choice of base model section

        random_forest_radio_btn = Radiobutton(self, text='Random forest', variable=self.ensemble_models_type,
                                              background=self.bg, value='random_forest',
                                              command=self.select_ensemble_model_without_choice)
        random_forest_radio_btn.grid(column=2, row=4, sticky='w')

        # Classic ensemble models section

        bagging_radio_btn = Radiobutton(self, text='Bagging', variable=self.ensemble_models_type,
                                        background=self.bg, value='bagging')
        bagging_radio_btn.grid(column=2, row=5, sticky='w')

        grad_boosting_btn = Radiobutton(self, text='Gradient boosting', variable=self.ensemble_models_type,
                                        background=self.bg, value='grad_boosting',
                                        command=self.select_ensemble_model_without_choice)
        grad_boosting_btn.grid(column=2, row=6, sticky='w')

        ada_boosting_btn = Radiobutton(self, text='AdaBoosting', variable=self.ensemble_models_type,
                                       background=self.bg, value='ada_boosting',
                                       command=self.select_ensemble_model_without_choice)
        ada_boosting_btn.grid(column=2, row=7, sticky='w')

        self.classic_models_fields = [
            select_model_label,
            select_classic_model_label,
            linear_regression_radio_btn,
            svm_radio_btn,
            tree_radio_btn,
            random_forest_radio_btn,
            select_ensemble_model_label,
            bagging_radio_btn,
            grad_boosting_btn,
            ada_boosting_btn
        ]
        self.drawing_dataset_and_predict_fields()

    def make_predict(self):
        #TODO
        #self.enter_dataset_field.get_text()
        if self.ensemble_models_type.get() == 'neural':
            pass

        if self.ensemble_models_type.get() == 'classic':
            pass

    def drawing_dataset_and_predict_fields(self):
        if self.predict_fields_was_drawn:
            return
        self.enter_dataset_field_label.grid(column=1, row=8, columnspan=2)
        self.enter_dataset_field.grid(column=1, row=9, columnspan=2)
        self.empty_row.grid(column=1, row=10, columnspan=2)
        make_predict_button = Button(self, text="Make predict", width=20,
                                     command=self.make_predict)
        make_predict_button.grid(column=1, row=11, columnspan=2)
        self.predict_fields_was_drawn = True


if __name__ == '__main__':
    window = App()
    window.mainloop()
