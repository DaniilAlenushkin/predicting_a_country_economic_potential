import pickle
from threading import Thread
from tkinter import *

import autokeras as ak
import keras.backend as k
import pandas as pd
from keras.models import load_model


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

        self.feed_forward_model = None
        self.gru_dense_model = None
        self.lstm_dense_model = None
        self.conv_gru_dense_model = None
        self.conv_lstm_dense_model = None

        # enter_dataset_section
        self.enter_dataset_field_label = Label(self, text='Enter dataset path:', background=self.bg)
        self.enter_dataset_field = Text(width=30, height=1, bg='white', fg='black', wrap=WORD)
        self.empty_row = Label(self, text='  ', background=self.bg)
        self.predict_fields_was_drawn = False
        self.predict_counter = 1
        self.info_label = Label(self, text='', background=self.bg)
        self.splitted_dataset = None

        # predict thred
        self.predict_thread = Thread(target=self.make_predict, daemon=True)

    def start_predict_thread(self):
        if self.predict_thread.is_alive():
            self.create_info_label('Wait for the previous predict to complete')
            return

        self.predict_thread = Thread(target=self.make_predict, daemon=True)
        self.predict_thread.start()

    def make_predict(self):
        self.create_info_label('Predict in progress...')

        def r2_score_custom(y_true, y_pred):
            ss_res = k.sum(k.square(y_true - y_pred))
            ss_tot = k.sum(k.square(y_true - k.mean(y_true)))
            r2 = 1 - ss_res / (ss_tot + k.epsilon())
            return r2

        def r2_loss(y_true, y_pred):
            return -r2_score_custom(y_true, y_pred)

        def save_predict(data, file_name):
            df = pd.DataFrame(data)
            file_name = file_name + f'_{self.predict_counter}' + '.xlsx'
            df.to_excel(file_name, index=False)
            self.create_info_label(f'Predict saved: {file_name}')
            self.predict_counter += 1

        def load_predict_and_save_pickle_model_predict(file_name, x_for_predict):
            with open(f'regression_models/{file_name}.pkl', 'rb') as f:
                model = pickle.load(f)
            save_predict(model.predict(x_for_predict), file_name)

        try:
            x_test = pd.read_excel(self.enter_dataset_field.get("1.0", END).strip('\n'))

            if self.model_type.get() == 'neural':
                if self.neural_models_type.get() == 'feed_forward':
                    if self.feed_forward_model is None:
                        self.feed_forward_model = load_model('automl/models_for_bfill_ffill/best_model',
                                                             custom_objects={"r2_loss": r2_loss,
                                                                             "r2_score_custom": r2_score_custom})
                    save_predict(self.feed_forward_model.predict(x_test), 'feed_forward_neural_network_predict')

                elif self.neural_models_type.get() == 'gru_dense':
                    if self.gru_dense_model is None:
                        self.gru_dense_model = load_model('automl/nm_for_bfill_ffill_gru_v2/best_model',
                                                          custom_objects=ak.CUSTOM_OBJECTS)
                    save_predict(self.gru_dense_model.predict(x_test), 'gru_dense_neural_network_predict')

                elif self.neural_models_type.get() == 'lstm_dense':
                    if self.lstm_dense_model is None:
                        self.lstm_dense_model = load_model('automl/nm_for_bfill_ffill_lstm_v2/best_model',
                                                           custom_objects=ak.CUSTOM_OBJECTS)
                    save_predict(self.lstm_dense_model.predict(x_test), 'lstm_dense_neural_network_predict')

                elif self.neural_models_type.get() == 'conv_gru_dense':
                    if self.conv_gru_dense_model is None:
                        self.conv_gru_dense_model = load_model('automl/nm_for_bfill_ffill_conv_gru/best_model',
                                                               custom_objects=ak.CUSTOM_OBJECTS)
                    save_predict(self.conv_gru_dense_model.predict(x_test), 'conv_gru_dense_neural_network_predict')

                elif self.neural_models_type.get() == 'conv_lstm_dense':
                    if self.conv_lstm_dense_model is None:
                        self.conv_lstm_dense_model = load_model('automl/nm_for_bfill_ffill_conv_lstm/best_model',
                                                                custom_objects=ak.CUSTOM_OBJECTS)
                    save_predict(self.conv_lstm_dense_model.predict(x_test), 'conv_lstm_dense_neural_network_predict')

                else:
                    self.create_info_label('Select neural model')

            if self.model_type.get() == 'classic':

                if self.ensemble_models_type.get() == 'bagging':
                    match self.classic_model_type.get():
                        case 'svm':
                            load_predict_and_save_pickle_model_predict('bagging_plus_svr_model', x_test)
                        case 'linear_regression':
                            load_predict_and_save_pickle_model_predict('bagging_plus_linear_model', x_test)
                        case 'decision_tree':
                            load_predict_and_save_pickle_model_predict('bagging_plus_decision_tree_model', x_test)
                        case _:
                            self.create_info_label('Select the model on which the bagging should be based')

                elif self.ensemble_models_type.get() == 'grad_boosting':
                    load_predict_and_save_pickle_model_predict('gradient_boosting_model', x_test)

                elif self.ensemble_models_type.get() == 'ada_boosting':
                    load_predict_and_save_pickle_model_predict('adaptive_boosting_model', x_test)

                elif self.ensemble_models_type.get() == 'random_forest':
                    load_predict_and_save_pickle_model_predict('random_forest_model', x_test)

                elif self.classic_model_type.get() == 'svm':
                    load_predict_and_save_pickle_model_predict('svr_model', x_test)

                elif self.classic_model_type.get() == 'linear_regression':
                    load_predict_and_save_pickle_model_predict('linear_regression_model', x_test)

                elif self.classic_model_type.get() == 'decision_tree':
                    load_predict_and_save_pickle_model_predict('decision_tree_regressor_model', x_test)

                else:
                    self.create_info_label('Select classic regression model')

        except Exception as e:
            self.create_info_label(str(e))

    def select_ensemble_model_without_choice(self):
        self.classic_model_type.set(self.NO_VALUE_SELECTED)

    def select_classic_regression_model(self):
        if self.ensemble_models_type.get() != 'bagging':
            self.ensemble_models_type.set(self.NO_VALUE_SELECTED)

    def select_neural_models(self):
        self.create_info_label('')
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
        self.create_info_label('')

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

    def create_info_label(self, text):
        self.info_label.config(text=text)
        self.info_label.grid(column=1, row=12, columnspan=2)

    def drawing_dataset_and_predict_fields(self):
        if self.predict_fields_was_drawn:
            return
        self.enter_dataset_field_label.grid(column=1, row=8, columnspan=2)
        self.enter_dataset_field.grid(column=1, row=9, columnspan=2)
        self.empty_row.grid(column=1, row=10, columnspan=2)
        make_predict_button = Button(self, text="Make predict", width=20,
                                     command=self.start_predict_thread)
        make_predict_button.grid(column=1, row=11, columnspan=2)
        self.predict_fields_was_drawn = True


if __name__ == '__main__':
    window = App()
    window.mainloop()
