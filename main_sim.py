import numpy as np
from keras.models import load_model
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


from deep_learning_models import remove_dense, adv_net, non_adv_net
from sklearn.metrics.pairwise import cosine_similarity
from dataset_preparation import awgn, LoadDataset, ChannelIndSpectrogram


from sklearn.manifold import TSNE

#%%

def edge_clf_awgn(
              file_path,
              clf_path,
              snr_synthetic,
              tx_range = np.arange(30,40,dtype = int), 
              pkt_range = np.arange(100,200,dtype = int)
              ):
    
    LoadDatasetObj = LoadDataset()
    
    data, true_label, _, _ = LoadDatasetObj.load_iq_samples(file_path, 
                                                            tx_range,  
                                                            pkt_range)
    
    data = awgn(data, [snr_synthetic])
    snr = np.ones([len(data),1])*snr_synthetic
    
    true_label = true_label - tx_range[0]
    
    ChannelIndSpectrogramObj = ChannelIndSpectrogram()

    # data = awgn(data, [60])

    data = ChannelIndSpectrogramObj.channel_ind_spectrogram(data)
    
    
    pred_prob = clf_net.predict(data)


    fuse_num = 1
    """Fuse multiple packets"""
    if fuse_num > 1:

        pred_prob_fused = pred_prob

        num_tx = len(tx_range)
        num_pkt = len(pkt_range)

        for dev_ind in range(num_tx):

            pred_per_dev = pred_prob[dev_ind*num_pkt:(dev_ind + 1)*num_pkt]

            pred_fused_per_dev = np.empty(pred_per_dev.shape)

            pred_fused_per_dev[0:fuse_num] = pred_per_dev[0:fuse_num]

            for pkt_ind in range(fuse_num, num_pkt):

                pred_fused_per_dev[pkt_ind] = np.mean(
                    pred_per_dev[pkt_ind-fuse_num:pkt_ind], axis=0)

            pred_prob_fused[dev_ind *
                            num_pkt:(dev_ind + 1)*num_pkt] = pred_fused_per_dev

        pred_prob = pred_prob_fused

    pred_label = pred_prob.argmax(axis=-1)
    
    
    print('Average SNR is ' + str(np.mean(snr)))
    
    acc = accuracy_score(true_label, pred_label)
    print('Overall accuracy = %.4f' % acc)
    
    
    plt_confusion_matrix = False
    if (plt_confusion_matrix):
        conf_mat = confusion_matrix(true_label, pred_label)
        classes = tx_range - tx_range[0] + 1
        
        plt.figure()
        sns.heatmap(conf_mat, annot=True, 
                    fmt = 'd', cmap='Blues',
                    cbar = False,
                    xticklabels=classes, 
                    yticklabels=classes)
        
        plt.xlabel('Predicted label', fontsize = 12)
        plt.ylabel('True label', fontsize = 12)
        # plt.savefig('confusion_matrix.pdf', bbox_inches='tight')
    
    return acc, pred_prob, pred_label, true_label, snr



def collaborative_clf_awgn(
                      files_path,
                      clf_net,
                      snr_all,
                      tx_range = np.arange(30,40,dtype = int), 
                      pkt_range_clf = np.arange(100,200,dtype = int),
                      ):
    
    def _cal_weights(snr_rx):
            
        snr_linear = 10**(np.array(snr_rx)/10)
        weights = snr_linear/sum(snr_linear)
            
        return weights
    
    
    cloud_label = []
    cloud_prob = []
    
    num_rx = len(files_path)
    num_pkt = len(pkt_range_clf)*len(tx_range)
    snr_rx = np.empty([num_pkt, num_rx])
    
    edge_acc_all = []
    
    for rx_idx in range(num_rx):
        
        file_name = files_path[rx_idx]
        snr_synthetic = snr_all[rx_idx]
        
        edge_acc, edge_prob, edge_label, true_label, snr =  edge_clf_awgn(  
                                                                         file_name,
                                                                         clf_net,
                                                                         snr_synthetic,
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
    
    
    cloud_prob_weighted = np.zeros(cloud_prob.shape)
    for pkt_idx in range(num_pkt):
        weights_pkt = _cal_weights(snr_rx[pkt_idx])
        cloud_prob_pkt = cloud_prob[:,pkt_idx,:]
        for rx_idx in range(num_rx):
            cloud_prob_weighted[rx_idx, pkt_idx, :] = weights_pkt[rx_idx]*cloud_prob_pkt[rx_idx]
    
    fusion_weighted = np.mean(cloud_prob_weighted,axis = 0)
    pred_label_weighted = fusion_weighted.argmax(axis=-1)
    collaborative_weighted_acc = accuracy_score(true_label, pred_label_weighted)
    
    fusion = np.mean(cloud_prob,axis = 0)
    pred_label = fusion.argmax(axis=-1)
    collaborative_acc = accuracy_score(true_label, pred_label) 
    
    print('\nOverall accuracy is ' + str(collaborative_acc))
    
    
    plt_confusion_matrix = False
    if (plt_confusion_matrix):
        conf_mat = confusion_matrix(true_label, pred_label)
        classes = tx_range - tx_range[0] + 1
        
        plt.figure()
        sns.heatmap(conf_mat, annot=True, 
                    fmt = 'd', cmap='Blues',
                    cbar = False,
                    xticklabels=classes, 
                    yticklabels=classes)
        
        plt.xlabel('Predicted label', fontsize = 12)
        plt.ylabel('True label', fontsize = 12)
    
    return collaborative_weighted_acc, collaborative_acc, edge_acc_all




if __name__ == '__main__':
    
    files_path = [
                # './dataset/h5py_file/Test/pluto_2_test.h5',
                # './dataset/h5py_file/Test/b200_2_test.h5',
                # './dataset/h5py_file/Test/b200_mini_2_test.h5',
                # './dataset/h5py_file/Test/b210_2_test.h5',
                './dataset/h5py_file/Test/n210_1_test.h5',
                './dataset/h5py_file/Test/n210_2_test.h5',
                './dataset/h5py_file/Test/n210_3_test.h5',
                 ]
    
    collaborative_weighted_acc_all = [] 
    collaborative_acc_all = []
    acc_edge = []
    for i in range(0, 41, 5):
        
        # snr_all = np.ones(len(files_path))*i
        snr_all = np.array([0, 10, i])
        # snr_all = np.array([0, 0, 0, 0, 0, 0, i]) 
        
        # clf_net = load_model('clf_128_diverse_5.h5', compile=False)
        clf_net = non_adv_net(datashape=None, num_classes=10)
        clf_net.load_weights('clf_128_diverse_5.h5')

        collaborative_weighted_acc, collaborative_acc, edge_acc_all = collaborative_clf_awgn(files_path, 
                                                                clf_net, 
                                                                snr_all,  
                                                                tx_range = np.arange(30,40,dtype = int),
                                                                pkt_range_clf = np.arange(100,200,dtype = int))
        
        collaborative_weighted_acc_all.append(collaborative_weighted_acc)
        collaborative_acc_all.append(collaborative_acc)
        acc_edge.append(edge_acc_all)
    
    acc_edge = np.array(acc_edge)
    # collaborative_acc_all = np.array(collaborative_acc_all)
    # collaborative_weighted_acc_all = np.array(collaborative_weighted_acc_all)
    print('edge:' + str(np.transpose(acc_edge)))
    # print('acc of n210 3:' +  str(acc_edge[:,2]))
    print('weighted:' + str(collaborative_weighted_acc_all))
    print('unweighted:' + str(collaborative_acc_all))

    

