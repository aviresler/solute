import comet_ml
from base.base_data_loader import BaseDataLoader
import csv
import numpy as np
import sys


class SoluteDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(SoluteDataLoader, self).__init__(config)

        c_solute, y_molar_c_mmol, y_mass_c_g100ml, y_room_temp,\
        y_room_rel_humid, spectra_arry = self.read_spetra_from_disk(self.config.data_base.path)

        c_solute, y_molar_c_mmol, y_mass_c_g100ml, y_room_temp, \
        y_room_rel_humid, spectra_arry = self.get_specific_solute_database(c_solute, y_molar_c_mmol, y_mass_c_g100ml, y_room_temp, \
                                                                           y_room_rel_humid, spectra_arry, self.config.data_base.solute)

        self.c_solute, self.y_molar_c_mmol, self.y_mass_c_g100ml, self.y_room_temp, \
        self.y_room_rel_humid, self.spectra_arry =  self.augment_and_shuffle_solute_database(c_solute, y_molar_c_mmol, y_mass_c_g100ml, y_room_temp, \
                                                                                   y_room_rel_humid, spectra_arry, 10,  np.mean(np.std(spectra_arry, axis=0)))

        print(self.spectra_arry.shape)

    def get_train_data(self):
        #print(np.expand_dims(self.spectra_arry, axis=2).shape)
        print(len(self.y_molar_c_mmol))
        return np.expand_dims(self.spectra_arry, axis=2), np.array(self.y_molar_c_mmol, dtype=np.float32)

    def get_test_data(self):
        return self.X_test, self.y_test

    def read_spetra_from_disk(self,text_file_path):
        with open(text_file_path, "r") as in_text:
            in_reader = csv.DictReader(in_text, delimiter='\t')
            c_solute = []
            y_molar_c_mmol = []
            y_mass_c_g100ml = []
            y_room_temp = []
            y_room_rel_humid = []
            for row in in_reader:
                c_solute.append(row['C_Solute'])
                y_molar_c_mmol.append(row['Y_Molar_concentration_mM'])
                y_mass_c_g100ml.append(row['Y_Mass_concentration_g100ml'])
                y_room_temp.append(row['Y_Room_temperature'])
                y_room_rel_humid.append(row['Y_Room_relHumidity'])

        with open(text_file_path, "r") as in_text:
            in_reader_raw = csv.reader(in_text, delimiter='\t')
            spectra = []
            count = 0
            for row in in_reader_raw:
                if count != 0:
                    spectra.append(row[8:])
                count = count + 1

            return c_solute, y_molar_c_mmol, y_mass_c_g100ml, y_room_temp, y_room_rel_humid, \
                   np.asarray(spectra, dtype=np.float32)

    def get_specific_solute_database(self, c_solute, y_molar_c_mmol, y_mass_c_g100ml, y_room_temp, \
                                     y_room_rel_humid, spectrums, solute_name):

        indices = [i for i, x in enumerate(c_solute) if x == solute_name]
        c_solute_ = [c_solute[i] for i in indices]
        y_molar_c_mmol_ = [y_molar_c_mmol[i] for i in indices]
        y_mass_c_g100ml_ = [y_mass_c_g100ml[i] for i in indices]
        y_room_temp_ = [y_room_temp[i] for i in indices]
        y_room_rel_humid_ = [y_room_rel_humid[i] for i in indices]
        spectrums_ = spectrums[indices, :]

        return c_solute_, y_molar_c_mmol_, y_mass_c_g100ml_, y_room_temp_, y_room_rel_humid_, spectrums_

    def augment_and_normalize_sample(self, sample_in, std_in, number_of_samples):
        # noramlize
        sample_in /= np.max(np.abs(sample_in))

        ind = np.linspace(0, sample_in.shape[0] - 1, sample_in.shape[0])
        out_matrix = np.zeros((number_of_samples, sample_in.shape[0]), dtype=np.float32)
        out_matrix[0, :] = sample_in
        rand_slope = 0.95 + 0.1 * np.random.rand(number_of_samples - 1)
        rand_offset = -0.1 * std_in + 0.2 * std_in * np.random.rand(number_of_samples - 1)
        rand_scale = 0.9 + 0.2 * std_in * np.random.rand(number_of_samples - 1)

        for m in range(number_of_samples - 1):
            p = np.polyfit(ind, sample_in, 1)
            poly_data = np.polyval(p, ind)
            residue = sample_in - poly_data
            p1 = [rand_slope[m] * p[0], rand_offset[m] + p[1]]
            poly_modified = np.polyval(p1, ind)
            new_sample = rand_scale[m] * (poly_modified + residue)
            out_matrix[m + 1, :] = new_sample

        # for m in range(number_of_samples):
        #   plt.plot(ind, out_matrix[m, :], color='blue', linewidth=2)
        # plt.show()
        return out_matrix

    def augment_and_shuffle_solute_database(self, c_solute, y_molar_c_mmol, y_mass_c_g100ml, y_room_temp, \
                                            y_room_rel_humid, spectrums, augment_ratio, std):

        spectrum_mat = np.zeros((augment_ratio * spectrums.shape[0], spectrums.shape[1]), dtype=np.float32)
        c_solute_ = []
        y_molar_c_mmol_ = []
        y_mass_c_g100ml_ = []
        y_room_temp_ = []
        y_room_rel_humid_ = []

        for k in range(spectrums.shape[0]):
            spectrum_mat[k * augment_ratio:(k + 1) * augment_ratio, :] = self.augment_and_normalize_sample(spectrums[k], std,
                                                                                                      augment_ratio)
            c_solute_ += augment_ratio * c_solute
            y_molar_c_mmol_ += augment_ratio * y_molar_c_mmol
            y_mass_c_g100ml_ += augment_ratio * y_mass_c_g100ml
            y_room_temp_ += augment_ratio * y_room_temp
            y_room_rel_humid_ += augment_ratio * y_room_rel_humid

        # shuffling
        permut = np.random.permutation(augment_ratio * spectrums.shape[0])
        c_solute_ = [c_solute_[i] for i in permut]
        y_molar_c_mmol_ = [y_molar_c_mmol_[i] for i in permut]
        y_mass_c_g100ml_ = [y_mass_c_g100ml_[i] for i in permut]
        y_room_temp_ = [y_room_temp_[i] for i in permut]
        y_room_rel_humid_ = [y_room_rel_humid_[i] for i in permut]
        spectrum_mat[:, :] = spectrum_mat[permut, :]
        return c_solute_, y_molar_c_mmol_, y_mass_c_g100ml_, y_room_temp_, y_room_rel_humid_, spectrum_mat
