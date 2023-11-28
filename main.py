import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# from tensorflow.keras.optimizers import SGD
import tensorflow as tf

from dataset_preparation import awgn, LoadDataset, ChannelIndSpectrogram, data_generator, data_generator_adversarial

from deep_learning_models import adv_net, non_adv_net, remove_rx_clf
from tensorflow.keras.utils import to_categorical


def advsarial_training(
        file_path,
        tx_range=np.arange(30, 40, dtype=int),
        pkt_range=np.arange(0, 800, dtype=int),
):
    LoadDatasetObj = LoadDataset()

    '''Load data and label'''
    if len(file_path) == 1:

        data, tx_label, _, _ = LoadDatasetObj.load_iq_samples(file_path[0],
                                                              tx_range,
                                                              pkt_range)
    else:
        data, tx_label, rx_label = LoadDatasetObj.load_multiple_rx_data(file_path,
                                                                        tx_range,
                                                                        pkt_range)

    '''Shuffle data and label'''
    index = np.arange(len(tx_label))
    np.random.shuffle(index)
    data = data[index, :]
    tx_label = tx_label[index]
    if len(file_path) != 1:
        rx_label = rx_label[index]

    tx_label = to_categorical(tx_label - tx_range[0])
    if len(file_path) != 1:
        rx_label = to_categorical(rx_label)

    num_train_samples = round(0.9*len(data))
    num_valid_samples = len(data) - num_train_samples

    batch_size = 64 

    '''Create neural network and data generator'''
    if len(file_path) == 1:
        model = non_adv_net(data.shape, len(tx_range))
        train_generator = data_generator(data[:num_train_samples], tx_label[:num_train_samples], batch_size = batch_size)
        valid_generator = data_generator(data[num_train_samples:], tx_label[num_train_samples:], batch_size = batch_size)
    else:
        model = adv_net(data.shape, len(tx_range), len(file_path))
        # model = adv_lstm(data.shape, len(tx_range), len(file_path))
        train_generator = data_generator_adversarial(data[:num_train_samples], tx_label[:num_train_samples], rx_label[:num_train_samples], batch_size = batch_size)
        valid_generator = data_generator_adversarial(data[num_train_samples:], tx_label[num_train_samples:], rx_label[:num_train_samples], batch_size = batch_size)

    '''Training configurations'''
    early_stop = EarlyStopping('val_loss', min_delta=0, patience=20)
    reduce_lr = ReduceLROnPlateau('val_loss', min_delta=0, factor=0.2, patience=10, verbose=1)
    callbacks = [early_stop, reduce_lr]

    opt = tf.keras.optimizers.SGD(learning_rate=1e-3, momentum=0.9)

    '''Training'''
    if len(file_path) == 1:
        model.compile(loss='categorical_crossentropy', optimizer=opt)

        history = model.fit(train_generator,
                    steps_per_epoch=num_train_samples // batch_size,
                    validation_data=valid_generator,
                    validation_steps=num_valid_samples // batch_size,
                    epochs=500,
                    verbose=1,
                    callbacks=callbacks)
    else:
        model.compile(loss='categorical_crossentropy',
                      loss_weights={'tx_classifier': 1, 'rx_classifier': 1},
                      optimizer=opt)
        
        history = model.fit(train_generator,
                    steps_per_epoch=num_train_samples // batch_size,
                    validation_data=valid_generator,
                    validation_steps=num_valid_samples // batch_size,
                    epochs=500,
                    verbose=1,
                    callbacks=callbacks)


        model = remove_rx_clf(model)

    return model


def fine_tune(
        file_path,
        clf_net,
        tx_range=np.arange(30, 40, dtype=int),
        pkt_range=np.arange(0, 20, dtype=int)
):
    
    '''Load fine-tuning data'''
    LoadDatasetObj = LoadDataset()
    data, tx_label, _, _ = LoadDatasetObj.load_iq_samples(file_path,
                                                       tx_range,
                                                       pkt_range)

    '''Shuffle data and label'''
    index = np.arange(len(tx_label))
    np.random.shuffle(index)
    data = data[index, :]
    tx_label = tx_label[index]

    tx_label = to_categorical(tx_label - tx_range[0])

    # snr_range = np.arange(0,40)  # define SNR range
    # data = awgn(data, snr_range)

    ChannelIndSpectrogramObj = ChannelIndSpectrogram()

    data = ChannelIndSpectrogramObj.channel_ind_spectrogram(data)

    '''Fine-tuning configurations'''
    opt = tf.keras.optimizers.SGD(learning_rate=1e-5, momentum=0.9)
   
    clf_net.compile(loss='categorical_crossentropy', optimizer=opt)

    history = clf_net.fit(data,
                          tx_label,
                          epochs=20,
                          shuffle=True,
                          verbose=1,
                          batch_size=32)

    return clf_net

def edge_clf(
        file_path,
        clf_net,
        tx_range=np.arange(30, 40, dtype=int),
        pkt_range=np.arange(100, 200, dtype=int)
):

    LoadDatasetObj = LoadDataset()

    data, true_label, snr, _ = LoadDatasetObj.load_iq_samples(file_path,
                                                           tx_range,
                                                           pkt_range)
    true_label = true_label - tx_range[0]

    ChannelIndSpectrogramObj = ChannelIndSpectrogram()

    # data = awgn(data, [60])

    data = ChannelIndSpectrogramObj.channel_ind_spectrogram(data)

    # sns.heatmap(data[0, :, :, 0], cmap='Blues', cbar=False,
                # xticklabels=False, yticklabels=False, ).invert_yaxis()
    # plt.savefig('spectrogram.pdf', bbox_inches='tight')

    # data = data[:,:,:,0]
    # data = data.transpose(0,2,1) # [samples, timesteps, features]
    
    pred_prob = clf_net.predict(data)


    """Fuse the predictions on multiple packets"""
    fuse_num = 5 # number of packets involved during packet fusion 
    
    if fuse_num > 1:

        pred_prob_fused = pred_prob

        num_tx = len(tx_range)
        num_pkt = len(pkt_range)

        for dev_ind in range(num_tx):

            pred_per_dev = pred_prob[dev_ind*num_pkt:(dev_ind + 1)*num_pkt]

            pred_fused_per_dev = np.zeros(pred_per_dev.shape)

            pred_fused_per_dev[0:fuse_num] = pred_per_dev[0:fuse_num]

            for pkt_ind in range(fuse_num, num_pkt):

                pred_fused_per_dev[pkt_ind] = np.mean(
                    pred_per_dev[pkt_ind-fuse_num:pkt_ind], axis=0)

            pred_prob_fused[dev_ind *
                            num_pkt:(dev_ind + 1)*num_pkt] = pred_fused_per_dev

        pred_prob = pred_prob_fused


    pred_label = pred_prob.argmax(axis=-1)

    # snr[snr > 1e308] = 40

    print('Average SNR is ' + str(np.mean(snr)))

    # pred_label = pred_label + 30

    acc = accuracy_score(true_label, pred_label)
    print('Overall accuracy = %.4f' % acc)

    plt_confusion_matrix = True
    if (plt_confusion_matrix):
        conf_mat = confusion_matrix(true_label, pred_label)
        classes = tx_range - tx_range[0] + 1

        plt.figure()
        sns.heatmap(conf_mat, annot=True,
                    fmt='d', cmap='Blues',
                    cbar=False,
                    xticklabels=classes,
                    yticklabels=classes)

        plt.xlabel('Predicted label', fontsize=12)
        plt.ylabel('True label', fontsize=12)
        # plt.savefig('confusion_matrix.pdf', bbox_inches='tight')

    return acc, pred_prob, pred_label, true_label, snr

def collaborative_clf(
        files_path,
        clf_net,
        tx_range=np.arange(30, 40, dtype=int),
        pkt_range_clf=np.arange(100, 200, dtype=int),
):
    def _cal_weights(snr_rx):

        snr_linear = 10 ** (np.array(snr_rx) / 10)
        weights = snr_linear / sum(snr_linear)

        return weights

    cloud_label = []
    cloud_prob = []

    num_rx = len(files_path)
    num_pkt = len(pkt_range_clf) * len(tx_range)
    snr_rx = np.empty([num_pkt, num_rx])

    edge_acc_all = []

    for rx_idx in range(num_rx):
        file_name = files_path[rx_idx]

        edge_acc, edge_prob, edge_label, true_label, snr = edge_clf(
            file_name,
            clf_net,
            tx_range,
            pkt_range_clf
        )

        snr_rx[:, [rx_idx]] = snr
        print('Average SNR of receiver ' + str(rx_idx) + ' is '
              + str(np.mean(snr)) + ' dB. Edge classification accuracy is '
              + str(edge_acc))

        cloud_label.append(edge_label)
        cloud_prob.append(edge_prob)
        edge_acc_all.append(edge_acc)

    cloud_label = np.array(cloud_label)
    cloud_prob = np.array(cloud_prob)

    cloud_prob_weighted = np.empty(cloud_prob.shape)
    for pkt_idx in range(num_pkt):
        weights_pkt = _cal_weights(snr_rx[pkt_idx])
        cloud_prob_pkt = cloud_prob[:, pkt_idx, :]
        for rx_idx in range(num_rx):
            cloud_prob_weighted[rx_idx, pkt_idx, :] = weights_pkt[rx_idx] * cloud_prob_pkt[rx_idx]

    fusion_weighted = np.mean(cloud_prob_weighted, axis=0)
    pred_label_weighted = fusion_weighted.argmax(axis=-1)
    adaptive_soft_acc = accuracy_score(true_label, pred_label_weighted)

    fusion = np.mean(cloud_prob, axis=0)
    pred_label = fusion.argmax(axis=-1)
    soft_acc = accuracy_score(true_label, pred_label)

    print('\nOverall accuracy is ' + str(soft_acc) + ', adaptive soft accuracy is ' + str(adaptive_soft_acc) + '.\n')

    plt_confusion_matrix = False
    if (plt_confusion_matrix):
        conf_mat = confusion_matrix(true_label, pred_label)
        classes = tx_range - tx_range[0] + 1

        plt.figure()
        sns.heatmap(conf_mat, annot=True,
                    fmt='d', cmap='Blues',
                    cbar=False,
                    xticklabels=classes,
                    yticklabels=classes)

        plt.xlabel('Predicted label', fontsize=12)
        plt.ylabel('True label', fontsize=12)

    return adaptive_soft_acc, soft_acc, edge_acc_all


if __name__ == '__main__':

    # Specifies what task the program runs for. 

    run_for = 'Train'
    # run_for = 'Edge Classification'
    # run_for = 'Fine Tune Before Classification'
    # run_for = 'Collaborative Classification'

    if run_for == 'Train':

        file_list = [
            './dataset/h5py_file/Train/rtl_1_train.h5',
            './dataset/h5py_file/Train/rtl_2_train.h5',
            './dataset/h5py_file/Train/rtl_3_train.h5',
            './dataset/h5py_file/Train/rtl_4_train.h5',
            './dataset/h5py_file/Train/rtl_5_train.h5',
            # './dataset/h5py_file/Train/pluto_1_train.h5',
            # './dataset/h5py_file/Train/b200_1_train.h5',
            # './dataset/h5py_file/Train/b200_mini_1_train.h5',
            # './dataset/h5py_file/Train/b210_1_train.h5'
        ]

        clf_net = advsarial_training(file_list)
        clf_net.save('b210_1_majie.h5')

        # file_list = [
        #             # './dataset/h5py_file/drift/Train/rtl_6_day1_wireless_train.h5',
        #             # './dataset/h5py_file/drift/Train/pluto_1_day1_wireless_train.h5',
        #             # './dataset/h5py_file/drift/Train/n210_1_day1_wireless_train.h5',
        #             './dataset/h5py_file/drift/Train/rtl_1_day1_wireless_train.h5',
        #             ] 

        # clf_net = advsarial_training(file_list)
        # # Save the trained model.
        # clf_net.save('day1_clf_128_rtl_1.h5')      

    elif run_for == 'Edge Classification':

        # dataset_path = './dataset/h5py_file/Test/n210_3_test.h5'
        # clf_net = load_model('cnn_hete_5.h5', compile=False)
        # acc = edge_clf(dataset_path, clf_net)

        file_list = [
        './dataset/h5py_file/Test/rtl_1_test.h5',
        # './dataset/h5py_file/Test/rtl_2_test.h5',
        # './dataset/h5py_file/Test/rtl_3_test.h5',
        # './dataset/h5py_file/Test/rtl_4_test.h5',
        # './dataset/h5py_file/Test/rtl_5_test.h5',
        # './dataset/h5py_file/Test/rtl_6_test.h5',
        # './dataset/h5py_file/Test/rtl_7_test.h5',
        # './dataset/h5py_file/Test/rtl_8_test.h5',
        # './dataset/h5py_file/Test/rtl_9_test.h5',
        './dataset/h5py_file/Test/pluto_1_test.h5',
        # './dataset/h5py_file/Test/pluto_2_test.h5',
        './dataset/h5py_file/Test/b200_1_test.h5',
        # './dataset/h5py_file/Test/b200_2_test.h5',
        './dataset/h5py_file/Test/b200_mini_1_test.h5',
        # './dataset/h5py_file/Test/b200_mini_2_test.h5',
        './dataset/h5py_file/Test/b210_1_test.h5',
        # './dataset/h5py_file/Test/b210_2_test.h5',
        './dataset/h5py_file/Test/n210_1_test.h5',
        # './dataset/h5py_file/Test/n210_2_test.h5',
        # './dataset/h5py_file/Test/n210_3_test.h5',
        # './dataset/h5py_file/drift/Test/rtl_6_day1_wireless_test.h5',
        # './dataset/h5py_file/drift/Test/rtl_6_day2_wireless_test.h5',
        # './dataset/h5py_file/drift/Test/rtl_6_day3_wireless_test.h5',
        # './dataset/h5py_file/drift/Test/rtl_6_day4_wireless_test.h5',
        # './dataset/h5py_file/drift/Test/n210_1_day1_wireless_test.h5',
        # './dataset/h5py_file/drift/Test/n210_1_day2_wireless_test.h5',
        # './dataset/h5py_file/drift/Test/n210_1_day3_wireless_test.h5',
        # './dataset/h5py_file/drift/Test/n210_1_day4_wireless_test.h5',
        ]

        clf_net = load_model('b210_1_majie.h5', compile=False)

        acc_all = []
        for dataset_path in file_list:
            acc = edge_clf(dataset_path, clf_net)
            acc_all.append(acc[0])

        print(acc_all)


    elif run_for == 'Fine Tune Before Classification':

        # file_list = [
        # './dataset/h5py_file/Test/rtl_1_test.h5',
        # './dataset/h5py_file/Test/rtl_2_test.h5',
        # './dataset/h5py_file/Test/rtl_3_test.h5',
        # './dataset/h5py_file/Test/rtl_4_test.h5',
        # './dataset/h5py_file/Test/rtl_5_test.h5',
        # './dataset/h5py_file/Test/rtl_6_test.h5',
        # './dataset/h5py_file/Test/rtl_7_test.h5',
        # './dataset/h5py_file/Test/rtl_8_test.h5',
        # './dataset/h5py_file/Test/rtl_9_test.h5',
        # # './dataset/h5py_file/Test/pluto_1_test.h5',
        # # './dataset/h5py_file/Test/pluto_2_test.h5',
        # # './dataset/h5py_file/Test/b200_1_test.h5',
        # # './dataset/h5py_file/Test/b200_2_test.h5',
        # # './dataset/h5py_file/Test/b200_mini_1_test.h5',
        # # './dataset/h5py_file/Test/b200_mini_2_test.h5',
        # # './dataset/h5py_file/Test/b210_1_test.h5',
        # # './dataset/h5py_file/Test/b210_2_test.h5',
        # # './dataset/h5py_file/Test/n210_1_test.h5',
        # # './dataset/h5py_file/Test/n210_2_test.h5',
        # # './dataset/h5py_file/Test/n210_3_test.h5'
        # ]

        # acc_all = []
        # for dataset_path in file_list:
        #     clf_net = load_model('cnn_hete_no_rtl.h5', compile=False)

        #     fine_tuned_clf_net = fine_tune(
        #                                     dataset_path, 
        #                                     clf_net, 
        #                                     tx_range = np.arange(30,40,dtype = int),
        #                                     pkt_range = np.arange(0,20, dtype = int)
        #                                     )

        #     acc = edge_clf(dataset_path, 
        #                     fine_tuned_clf_net,
        #                     tx_range = np.arange(30,40,dtype = int), 
        #                     pkt_range = np.arange(100,200,dtype = int))
        #     acc_all.append(acc[0])
        # print(acc_all)

        dataset_path = './dataset/h5py_file/Test/rtl_6_test.h5'
        acc_pkt = []
        for i in range(0, 101, 10):
            clf_net = load_model('cnn_hete_5.h5', compile=False)

            if i != 0:
                fine_tuned_clf_net = fine_tune(
                    dataset_path,
                    clf_net,
                    tx_range=np.arange(30, 40, dtype=int),
                    pkt_range=np.arange(0, i, dtype=int)
                )
            else:
                fine_tuned_clf_net = clf_net

            acc = edge_clf(dataset_path,
                           fine_tuned_clf_net,
                           tx_range=np.arange(30, 40, dtype=int),
                           pkt_range=np.arange(100, 200, dtype=int))

            acc_pkt.append(acc[0])

        print(acc_pkt)



    elif run_for == 'Collaborative Classification':

        clf_files = [
                    './dataset/h5py_file/Test/Location_A_n210_1_test.h5' ,
                    './dataset/h5py_file/Test/Location_A_n210_2_test.h5',
                    './dataset/h5py_file/Test/Location_A_n210_3_test.h5',
                    ]
        
        # clf_net = load_model('cnn_hete_5.h5', compile=False)
        clf_net = non_adv_net(datashape=None, num_classes=10)
        clf_net.load_weights('cnn_hete_5.h5')
        print('Load model done.')

        adaptive_soft_acc, soft_acc, edge_acc_all = collaborative_clf(clf_files,
                                                                      clf_net,
                                                                      tx_range=np.arange(30, 40, dtype=int),
                                                                      pkt_range_clf=np.arange(0, 300, dtype=int))
