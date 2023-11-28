import numpy as np
import h5py
from numpy import sum,sqrt
from numpy.random import standard_normal, uniform

from scipy import signal
import math 
import random
from natsort import natsorted


def awgn(data, snr_range):
    
    pkt_num = data.shape[0]
    SNRdB = uniform(snr_range[0],snr_range[-1],pkt_num)
    SNR_linear = 10**(SNRdB/10)
    for pktIdx in range(pkt_num):
        s = data[pktIdx]
        # SNRdB = uniform(snr_range[0],snr_range[-1])
        # SNR_linear = 10**(SNRdB[pktIdx]/10)
        P= sum(abs(s)**2)/len(s)
        N0=P/SNR_linear[pktIdx]
        n = sqrt(N0/2)*(standard_normal(len(s))+1j*standard_normal(len(s)))
        
        if np.isnan(np.sum(n)) or np.isinf(np.sum(n)):
            data[pktIdx] = s
        else:
            data[pktIdx] = s + n

    return data 


class LoadDataset():
    def __init__(self,):
        self.dataset_name = 'data'
        self.labelset_name = 'label'
        
    def _convert_to_complex(self, data):
        '''Convert the loaded data to complex IQ samples.'''
        num_row = data.shape[0]
        num_col = data.shape[1] 
        data_complex = np.zeros([num_row,round(num_col/2)],dtype=complex)
     
        data_complex = data[:,:round(num_col/2)] + 1j*data[:,round(num_col/2):] 
        return data_complex
    
    
    def load_iq_samples(self, file_path, dev_range, pkt_range):
        '''
        Load IQ samples from a dataset.
        
        INPUT:
            FILE_PATH is the dataset path.
            
            DEV_RANGE specifies the loaded device range.
            
            PKT_RANGE specifies the loaded packets range.
            
        RETURN:
            DATA is the laoded complex IQ samples.
            
            LABLE is the true label of each received packet.
        '''
        
        f = h5py.File(file_path,'r')
        label = f[self.labelset_name][:]
        label = label.astype(int)
        label = np.transpose(label)
        label = label - 1
        
        # if len(f.keys()) == 4:
        snr = f['SNR'][:]
        snr = np.transpose(snr)

        cfo = f['CFO'][:]
        cfo = np.transpose(cfo)
        # elif len(f.keys()) == 3:
        #     print('Warning: SNR information is not included in the dataset.')
        #     snr = np.ones((1,len(label)))*(-999)
        #     snr = np.transpose(snr)
            
        label_start = int(label[0]) + 1
        label_end = int(label[-1]) + 1
        num_dev = label_end - label_start + 1
        num_pkt = len(label)
        num_pkt_per_dev = int(num_pkt/num_dev)
        
        print('Dataset information: Dev ' + str(label_start) + ' to Dev ' + 
              str(label_end) + ', ' + str(num_pkt_per_dev) + ' packets per device.')
        
        sample_index_list = []
        
        for dev_idx in dev_range:
            sample_index_dev = np.where(label==dev_idx)[0][pkt_range].tolist()
            sample_index_list.extend(sample_index_dev)
    
        data = f[self.dataset_name][sample_index_list]
        data = self._convert_to_complex(data)
        
        label = label[sample_index_list]
        snr = snr[sample_index_list]
        cfo = cfo[sample_index_list]
        
        f.close()
        return data, label, snr, cfo
    
    
    def load_multiple_rx_data(self, file_list, tx_range, pkt_range):
        
        num_rx = len(file_list)
        num_tx = len(tx_range)
        num_pkt = len(pkt_range)
        
        data = []
        tx_label = []
        rx_label = []
        
        for file_idx in range(num_rx):
            print('Start loading dataset ' + str(file_idx + 1))
            filename = file_list[file_idx]
            # filename = folder_name + filename
            [data_temp, tx_label_temp, _, _ ] = self.load_iq_samples(filename, tx_range, pkt_range)
            rx_label_temp = np.ones(num_pkt*num_tx)*file_idx
            
            data.extend(data_temp)
            tx_label.extend(tx_label_temp)
            rx_label.extend(rx_label_temp)
            
        data = np.array(data)   
        tx_label = np.array(tx_label)
        rx_label = np.array(rx_label)
        
        return data, tx_label, rx_label



class ChannelIndSpectrogram():
    def __init__(self,):
        pass
    
    def _normalization(self,data):
        ''' Normalize the signal.'''
        s_norm = np.zeros(data.shape, dtype=complex)
        
        for i in range(data.shape[0]):
        
            sig_amplitude = np.abs(data[i])
            rms = np.sqrt(np.mean(sig_amplitude**2))
            s_norm[i] = data[i]/rms
        
        return s_norm        



    def _gen_single_channel_ind_spectrogram(self, sig, win_len, overlap):
        f, t, spec = signal.stft(sig, 
                                window='boxcar', # boxcar
                                nperseg= win_len, 
                                noverlap= overlap, 
                                nfft= win_len,
                                return_onesided=False, 
                                padded = False, 
                                boundary = None)
        
        spec = np.fft.fftshift(spec, axes=0)
        spec = spec + 1e-12
        
        dspec = spec[:,1:]/spec[:,:-1]    
                 
        dspec_amp = np.log10(np.abs(dspec)**2)
                  
        return dspec_amp
    

    def channel_ind_spectrogram(self, data, win_len = 128, crop_ratio = 0.3):
        data = self._normalization(data)
        
        # win_len = 16
        overlap = round(0.5*win_len)
        
        num_sample = data.shape[0]
        # num_row = math.ceil(win_len*(1-2*crop_ratio))
        num_row = len(range(math.floor(win_len*crop_ratio),math.ceil(win_len*(1-crop_ratio))))
        num_column = int(np.floor((data.shape[1]-win_len)/(win_len - overlap)) + 1) - 1
         
        
        data_dspec = np.zeros([num_sample, num_row, num_column, 1])
        # data_dspec = []
        for i in range(num_sample):
                   
            dspec_amp = self._gen_single_channel_ind_spectrogram(data[i], win_len, overlap)
            dspec_amp = self._spec_crop(dspec_amp, crop_ratio)
            # dspec_phase = self._spec_crop(dspec_phase, crop_ratio)
            data_dspec[i,:,:,0] = dspec_amp
            # data_dspec[i,:,:,1] = dspec_phase
            
        return data_dspec   

    def _spec_crop(self, x, crop_ratio):
      
        num_row = x.shape[0]
        x_cropped = x[math.floor(num_row*crop_ratio):math.ceil(num_row*(1-crop_ratio))]
     
        return x_cropped
    
    


def data_generator(data_source, label_source, batch_size, snr_range=range(0, 50)):

    ChannelIndSpectrogramObj = ChannelIndSpectrogram()
    
    while True:
        
        sample_ind = random.sample(range(0, len(data_source)), batch_size)
        
        data = data_source[sample_ind]
        data = awgn(data, snr_range)
        data = ChannelIndSpectrogramObj.channel_ind_spectrogram(data)
        
        label = label_source[sample_ind]
        
        yield data, label  

def data_generator_adversarial(data_source, label_source, rx_label_source, batch_size, snr_range=range(0, 50)):

    ChannelIndSpectrogramObj = ChannelIndSpectrogram()
    
    while True:
        
        sample_ind = random.sample(range(0, len(data_source)), batch_size)
        
        data = data_source[sample_ind]
        data = awgn(data, snr_range)

        data = ChannelIndSpectrogramObj.channel_ind_spectrogram(data)


        label =  label_source[sample_ind]
        label_rx = rx_label_source[sample_ind]
        
        yield data, [label, label_rx]