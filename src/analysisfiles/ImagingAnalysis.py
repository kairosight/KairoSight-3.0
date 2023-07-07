# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 11:07:44 2022

@author: Analysis ALPHA
"""

import copy
import csv
import cv2
import matplotlib.cm as cm 
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from numpy import amax, amin, inf, load, loadtxt, save, savetxt
import numpy as np
import os
#import pandas as pd
from PyQt5.QtWidgets import QFileDialog
import tkinter
from tkinter import messagebox
from scipy.signal import find_peaks
from util.analysis import calc_tran_activation, draw_poly
from util.processing import normalize


class ImagingAnalysis: 
    def adjust_alternan(self, fps, img, li1, li2, transp, start_ind, end_ind, 
                      interp_selection, peak_coeff, file_id):
        #this is an apd map, the actual alternan map code needs to be called here
        imgff=normalize(img,start_ind,end_ind)
        midP = self.midpoint(imgff,peak_coeff)[0]
        
        self.imaging_mapping(midP*100, li1, li2, transp)
        #grabs the selected coordinates from the user, returns [[x y]]
        mid_pts = np.asarray(plt.ginput(n=1, timeout=-1, show_clicks=True))

        #if the user selects a point, close the window
        if mid_pts is not None:
            plt.close()
        #grabbing the x, y coordinates selected and getting the corresponding 
        #z value
        x=round(mid_pts[0][0]).astype(int)
        y=round(mid_pts[0][1]).astype(int)
        z=round((1-midP[y,x])*100).astype(int)
        mid=round((midP[y,x])*100).astype(int)
        print("Z Value:", z)
        print("Midpoint:", mid)
        print("X-Axis:", x)
        print("Y-Axis:", y)
        
    def alternan_50(self, fps, img, li1, li2, transp, start_ind, end_ind, 
                start_ind2, end_ind2, interp_selection, perc_apd, 
                image_type, file_id):
        
        if image_type == 0:
            mapp_trace1 = self.apd_analysis(fps, img, start_ind, end_ind, 
                                     interp_selection, perc_apd, file_id)[0]
            save('apd50_alt_1', mapp_trace1)
            mapp_trace2 = self.apd_analysis(fps, img, start_ind2, end_ind2, 
                                     interp_selection, perc_apd, file_id)[0]
            save('apd50_alt_2', mapp_trace2)
        elif image_type == 1:
            mapp_trace1 = self.amplitude_analysis()
            mapp_trace2 = self.amplitude_analysis()
            
        #getting the absolute of the difference between both alternan mapps
        numerator = abs(mapp_trace1 - mapp_trace2)
        #getting the average of the traces for both alternan mapps
        denominator = (mapp_trace1 + mapp_trace2)/2
        #calculating the alternan coefficients of both mapps
        alternan_coefficient = numerator/denominator
        #plotting the alternan coefficients
        self.imaging_mapping(alternan_coefficient*100, li1, li2, transp)
        
        #making a folder if there isn't a "Saved Data Maps" folder
        if not os.path.exists("Saved Data Maps"):
               os.makedirs("Saved Data Maps")
               
        #saving the data file
        savetxt('Saved Data Maps/AAC'+str(perc_apd*100)+'.csv', 
                alternan_coefficient, delimiter=',')
    
    def amplitude_analysis(self):
        #this is placeholder code for the alternan map amplitude calculation
        #needs to return a calc_tran_activation map
        amplitude = 1
        return amplitude
    
    #creating a function to analyze the apd80 of a trace
    def apd_analysis(self, fps, img, start_ind, end_ind, interp_selection, 
                    perc_apd, file_id): 
        #editing the image to use only the start and end tiff points
        imgf=copy.deepcopy(img)
        imgf=normalize(imgf,start_ind,end_ind)
        
        #if the value of the image is not a number, assign it zero 
        imgf[np.isnan(imgf)] = 0 
        
        #getting the shape of the image (x,y,z)
        aa=np.shape(imgf)
        #assigning the shape to an array
        aa=np.array(aa)
       
        #interp of zero is returning negative APDs, so this still needs fixed
        if interp_selection == 0:
            interp_selection = interp_selection + 1
            
            #create an empty array of zeros 
            mapp=np.zeros((aa[1],aa[2]))  
            #calculating the activation map
            act_ind = calc_tran_activation(imgf,0,int(aa[0]*0.55))
            
            min_inx=0
            
            #loop through the image array (aa), looking for APD values between
            #0.4 and 0.5, assign them to the empty array mapp
            for i in range(aa[2]):
                for j in range(aa[1]):
                    start=np.argmax(imgf[act_ind[j,i]:int(aa[0]*0.75),j,i])
                    start=act_ind[j,i]+start 
                    end=np.argmin(imgf[start:aa[0],j,i])
                    end=end+start
                    win1=1000
                    for k in range(start,end):
                        if imgf[k,j,i] > (((1-perc_apd) - 0.05) 
                                          and imgf[k,j,i] < 
                                          ((1-perc_apd) + 0.05)):
                            win=abs((1-perc_apd)-imgf[k,j,i])
                            if win<win1:
                                win1=win
                                min_inx=k
                                mapp[j,i]=min_inx
        else:
            interp_selection = interp_selection + 1

            #create an empty array of zeros 
            mapp=np.zeros((aa[1],aa[2]))  
            
          
            x_ori=np.linspace(0, aa[0], num=aa[0])
            x=np.linspace(0, aa[0], num=aa[0]*interp_selection)
            
            #create an empty array of zeros
            imgf_2=np.zeros((aa[0]*interp_selection,aa[1],aa[2]))
            
            #loop through the image array (aa), interpolating its values, and
            #assign these values to the empty array imgf_2
            for i in range(aa[2]):
                for j in range(aa[1]):
                    imgf_2[:,j,i] = np.interp(x, x_ori, imgf[:,j,i])

            aa2=np.shape(imgf_2)
            aa2=np.array(aa2)     
            
            #getting the activation map of the interpolated image (imgf_2)
            act_ind = calc_tran_activation(imgf_2, 0, int(aa2[0]*0.55))

            #loop through the image array (aa), looking for APD values between
            #0.4 and 0.5, assign them to the empty array mapp
            for i in range(aa[2]):
                for j in range(aa[1]): 
                    start=np.argmax(imgf_2[act_ind[j,i]:int(aa2[0]*0.75),j,i])
                    start=act_ind[j,i]+start
                    end=np.argmin(imgf_2[start:aa[0]*interp_selection,j,i])
                    end=end+start
                    win1=1000
                    for k in range(start,end):
                        if imgf_2[k,j,i] > (((1-perc_apd) - 0.05) 
                                            and imgf_2[k,j,i] < 
                                            ((1-perc_apd) + 0.05)):
                            win=abs((1-perc_apd)-imgf_2[k,j,i]) 
                            if win<win1:
                                win1=win
                                min_inx=k
                                mapp[j,i]=min_inx
                      
        #calculating the adjust action potentials by subtracting the values
        #from APD calculations at mapp from the action potentials of imgf_2
        mapp2=((1000/fps)/interp_selection)*(mapp-act_ind)
        mapp=((1000/fps)/interp_selection)*mapp
        
        #making a folder if there isn't a "Saved Data Maps" folder
        if not os.path.exists("Saved Data Maps\\" + file_id):
               os.makedirs("Saved Data Maps\\" + file_id)
               
        #saving the data file
        savetxt('Saved Data Maps\\' + file_id + '\\apd'+str(perc_apd*100)+'.csv', 
                mapp2, delimiter=',')
        
        return mapp2,mapp
 
    def ec_coupling_map_act(self,li1, li2, transp, start, end, interp, file_id):
        volt=load('filtered_voltage_image.npy')
        cal=load('filtered_calcium_image.npy')
         
        act_vm=calc_tran_activation(volt,start,end)
        act_ca=calc_tran_activation(cal,start,end)
        act_delay_map=act_ca-act_vm
        self.imaging_mapping(act_delay_map, li1, li2, transp)
        
        #making a folder if there isn't a "Saved Data Maps" folder
        if not os.path.exists("Saved Data Maps\\" + file_id):
               os.makedirs("Saved Data Maps\\" + file_id)
               
        #saving the data file
        savetxt('Saved Data Maps\\' + file_id + '\\activation_ACC.csv', 
                act_delay_map, delimiter=',')

 
    def ec_coupling_map_rep(self,fps,li1, li2, transp, start, end, 
                            apd_perc, interp, file_id):
        volt=load('filtered_voltage_image.npy')
        cal=load('filtered_calcium_image.npy')
       
        apd= self.apd_analysis(fps, volt, start, end, interp, apd_perc, file_id)[0]     
        cad= self.apd_analysis(fps, cal, start, end, interp, apd_perc, file_id)[0] 
        ap_ca_latency=cad-apd
        self.imaging_mapping(ap_ca_latency, li1, li2, transp) 
        
        #making a folder if there isn't a "Saved Data Maps" folder
        if not os.path.exists("Saved Data Maps\\" + file_id):
               os.makedirs("Saved Data Maps\\" + file_id)
               
        #saving the data file
        savetxt('Saved Data Maps\\' + file_id + '\\Repolarization_ACC'+ 
                str(apd_perc*100)+'.csv', ap_ca_latency, delimiter=',')  

    def ec_coupling_save(self, data_filt, image_drop):
        #creating a placeholder for the EC function for now
        if image_drop == 0:
            #save data_filt as voltage
            save('filtered_voltage_image', data_filt)
            # savetxt('filtered_voltage_image.csv', data_filt, delimiter=',')
        elif image_drop == 1:
            #save data_filt as calcium
            save('filtered_calcium_image', data_filt)
        
    def ec_coupling_signal_load(self):
        #setting the saved information to variables so they can be called later
        filtered_voltage = load('filtered_voltage_image.npy')
        filtered_calcium = load('filtered_calcium_image.npy')
        return filtered_voltage, filtered_calcium
    
    def interpol (self,imgf,interp_selection):   
        aa=np.shape(imgf)
        aa=np.array(aa)
        x_ori=np.linspace(0, aa[0], num=aa[0])
        x=np.linspace(0, aa[0], num=aa[0]*interp_selection)

        
        imgf_2=np.zeros((aa[0]*interp_selection,aa[1],aa[2]))
            
        for i in range(aa[2]):
            for j in range(aa[1]):
                imgf_2[:,j,i] = np.interp(x, x_ori, imgf[:,j,i])

        return imgf_2 
        
    def imaging_mapping(self, mapp2, li1, li2, transp): 
        mapp2[np.isnan(mapp2)] = 0 
        #setting a min apd value
        min_apd = float(amin(mapp2))
        if min_apd < 0:
            min_apd = 0
        #setting a max apd value
        max_apd = float(amax(mapp2))

        #setting the transp value to be a float
        transp = transp.astype(float)
        
        #if the user does not have a preset min and max apd then plot using 
        #these settings
        if li1 == "no preset min" and li2 == "no preset max":
            #show the plot of mapp2 ranging from min apd (0) to the max apd 
            #(80)
            plt.imshow(mapp2, alpha=transp, vmin=min_apd, vmax=max_apd, 
                        cmap='jet')
            #show the color bar to give reference to what the colors mean
            plt.colorbar(cm.ScalarMappable(colors.Normalize(min_apd, max_apd),
                                           cmap ='jet'),
                         cax = plt.axes([0.87, 0.1, 0.05, 0.8]),
                         format='%.3f')

        #if the user has a preset min and max apd, then plot using these 
        #settings and their preset values
        else:
            #assign min_apd to be what the user inputed
            min_apd = float(li1)
            #assign max_apd to be what the user inputed
            max_apd = float(li2)
            #show the plot of mapp2 ranging from 0 to the max apd (flot(li2))
            plt.imshow(mapp2, alpha=transp, vmin=min_apd, vmax=max_apd, 
                        cmap='jet')
            #show the color bar to give reference to what the colors mean
            plt.colorbar(cm.ScalarMappable(colors.Normalize(min_apd, max_apd), 
                                           cmap ='jet'),
                         cax = plt.axes([0.87, 0.1, 0.05, 0.8]), 
                         format='%.3f')
            
    #creating a function to generate and excel doc with the means of the post analysis
    def mean_post_analysis(self, file_id):
        #assigning the working directory to a variable
        directory = os.getcwd()
        
        #adding the Saved Data Maps to the working directory path
        ROI_analysis_folder = str(directory) + '\\ROI Analysis\\' + file_id
        
        #checking if a ROI folder exists, if not tell the user to use
        #region of interest analysis first
        if not os.path.exists(ROI_analysis_folder):
            window = tkinter.Tk()
            window.wm_withdraw()
            message = ("""\nThe 'Region of Interest Analysis' tool must be used before the 'Individual ROI Results' tool can be used. 
                  \n1. Select Region of Interest Analysis to analyze generated maps.
                  \n2. Then reselect Individual ROI Results.""")
            messagebox.showinfo(title="Warning", message=message)
        #if the ROI folder exists, continue
        else:
            rowcount = 0
            for row in open("ROI Analysis\\" + file_id + '\\all_results.csv'):
                rowcount += 1
    
            all_results = loadtxt("ROI Analysis\\" + file_id + '\\all_results.csv',  dtype='str', delimiter=',')
            
            #creating a variable so we can save the inputed number of regions
            region_number = tkinter.Tk()
            #closing the weird tkinter window that pops up
            region_number.wm_withdraw()
            
            #creating a dialogue box so the user can select the number of regions
            result_option = tkinter.simpledialog.askstring(
                'Result Option', 
                """\nType the result you want to save and select 'OK'.
                \n For Mean - type: mean
                \n For Median - type: median
                \n For SD - type: sd
                \n For N - type: n""", 
                parent = region_number)
            #pre-emptively removing spaces, in case the user types them
            result_option = result_option.replace(' ', '')
            
            #making sure the user didn't enter an invalid value
            if (result_option.lower() != 'mean' and result_option.lower() != 'median' and 
                result_option.lower() != 'sd' and result_option.lower() != 'n'):
                    window = tkinter.Tk()
                    window.wm_withdraw()
                    message = ("""\nAn invalid Result Option was selected.
                               \nPlease Reselect Individual ROI Results and enter 
                               a valid result option.""")
                    messagebox.showinfo(title="Warning", message=message)
                    
            #if a valid value was entered, continue with the organization
            else:
                #checking if the file path exists, if it does there is already a header
                #if it does not exist, a header hasn't been added yet
                if not os.path.exists("ROI Analysis\\" + file_id + "\\" + 
                                  result_option.lower() + "_individual_results.csv"):
                    header_exists = False
                else:
                    header_exists = True
                    
                with open("ROI Analysis\\" + file_id + "\\" + 
                          result_option.lower() + "_individual_results.csv", 'a') as file:                 
                    
                    #if there is no header add one, otherwise don't do anything
                    if header_exists == False:
                        header = ("Data File Name and Type,Region 1,Region 2,Region 3," +
                                  "Region 4,Region 5,Region 6,Region 7,Region 8," +
                                  "Region 9,Region 10")
                        
                        file.write(header + "\n")
                        file.close
                    else:
                        file.close
                        
                    file = open("ROI Analysis\\" + file_id + "\\" + 
                                result_option.lower() + "_individual_results.csv", 'a')
                    
                    i = 0
                    #stepping through all the rows in the csv to organize that data
                    while i < rowcount:
                        #checking if the row is a header and not with data
                        if all_results[i, 1] == 'mean':
                            #save the name to the file variable
                            data_file_name = all_results[i,0]
                            data_file_name = data_file_name.replace('_region', '')
                            
                            data = []
                            data.append(data_file_name)
                            
                            i += 1
                            #search the rest of the rows until a row with headers
                            #is reached or all the rows in the csv have been stepped
                            #through
                            while i < rowcount and all_results[i,1] != 'mean':
                                #getting the current region numbr
                                region_number = int(float(all_results[i,0]))
                                #if the value of the region number does not equal
                                #the length of the data array, then the user has 
                                #inputed specific region numbers and skipped the 
                                #current i region. So we want to skip that value 
                                #and just make it blank. 
                                if region_number != (len(data)):
                                    while len(data) < region_number:
                                        data.append(' ')
                                #then we can continue adding the value from the 
                                #region the user wants
                                if result_option.lower() == 'mean':
                                    data.append(all_results[i, 1])
                                elif result_option.lower() == 'median':
                                    data.append(all_results[i, 2])
                                elif result_option.lower() == 'sd':
                                    data.append(all_results[i, 3])
                                elif result_option.lower() == 'n':
                                    data.append(all_results[i, 4])
                                i += 1
                            
                            #creating a file object
                            writer = csv.writer(file)
                            #writing to the existing object (file)
                            writer.writerow(data)
                file.close()
    
    def midpoint(self,imgff,peak_coeff):
           aa=np.shape(imgff)
           aa=np.array(aa)
           peak1=np.zeros((aa[1],aa[2]))
           peak2=np.zeros((aa[1],aa[2]))

           for i in range(aa[1]):
               for j in range(aa[2]):
                   hh, _ = find_peaks(imgff[:,i,j], height=0.4,
                                      width=peak_coeff)
                   hh=hh.astype(int)
                   hh_size=np.shape(hh)
                   hh2=int(hh_size[0])
                   if hh2 > 1 and hh2 <3:
                       peak1[i,j]=hh[0]
                       peak2[i,j]=hh[1]
                   if hh2 > 2:   
                       if hh[1]-hh[0] < 8:  
                           peak1[i,j]=hh[0]
                           peak2[i,j]=hh[2]
                       else: 
                           peak1[i,j]=hh[0]
                           peak2[i,j]=hh[1]

           peak1=peak1.astype(int)
           peak2=peak2.astype(int)
         
           midP=np.zeros((aa[1],aa[2]))
           midP_idx=np.zeros((aa[1],aa[2]))
                
           for i in range(aa[1]):
               for j in range(aa[2]):
                   if peak2[i,j]>peak1[i,j] and peak2[i,j]>0:
                       midP[i,j]=np.amin(imgff[peak1[i,j]:peak2[i,j],i,j])
                       midP_idx[i,j]=np.argmin(imgff[peak1[i,j]:peak2[i,j],i,j])
           
           midP_idx=midP_idx+peak1
           
           return midP,midP_idx,peak1, peak2   
    
    def moving_alternan(self, fps, img, li1, li2, transp, start_ind, end_ind, 
                        image_type, peak_coeff, file_id):      
        if image_type == 0:
           
           imgff=copy.deepcopy(img)
           
           imgff=normalize(imgff,start_ind,end_ind)
           aa=np.shape(imgff)
           aa=np.array(aa)
           min_inx=0
           second_min=np.zeros((aa[1],aa[2]))
           second_min_idx=np.zeros((aa[1],aa[2]))
           apd1=np.zeros((aa[1],aa[2]))
           apd2=np.zeros((aa[1],aa[2]))
           midP, midP_idx, peak1, peak2 =self.midpoint(imgff,peak_coeff)
           
           for i in range(aa[1]):
               for j in range(aa[2]):
                   if peak2[i,j]>peak1[i,j] and peak2[i,j]>0:
                       
                       second_min[i,j]=np.round(np.amin(imgff[peak2[i,j]:
                                                              aa[0],i,j]), 2)
                       second_min_idx[i,j]=np.argmin((imgff[peak2[i,j]:
                                                           aa[0],i,j]) 
                                                     + peak2[i,j])
                       min_point=np.min(imgff[0:peak1[i,j],i,j])
                        
           ap_coeff=np.zeros((aa[1],aa[2])) 
           print(midP[36,34])
           print(midP_idx[36,34])
           
           for i in range(aa[1]):
               for j in range(aa[2]):
                   if peak2[i,j]>peak1[i,j] and peak2[i,j]>0: 
                       if  midP[i,j]>second_min[i,j]:
                           # duration calculation first beat 
                           start1=np.argmin(imgff[0:peak1[i,j],i,j])
                           end1=peak1[i,j]
                           dvdtmax=np.argmax(np.diff(imgff[start1:
                                                           end1,i,j])) + start1
                           apd1[i,j]=(1000/fps)*(midP_idx[i,j]-dvdtmax)

                           dvdtmax2=np.argmax(np.diff(
                               imgff[midP_idx[i,j].astype(int):
                                     peak2[i,j],i,j])) + midP_idx[i,j]
          
                           mid_amp=np.round(midP[i,j],2)
                           mid_amp1=mid_amp+0.2
                           mid_amp2=mid_amp-0.05
                           win1=1000
                           min_inx=0
                           
                           for k in range(peak2[i,j], (aa[0]*0.95).astype(int)):
                               if imgff[k,i,j] > (mid_amp2 and imgff[k,i,j] < 
                                                  mid_amp1): 
                                   win=abs(mid_amp-imgff[k,i,j])
                                   if win<win1:
                                       win1=win
                                       min_inx=k
                           
                           apd2[i,j]=(1000/fps)*(min_inx-dvdtmax2)
                           numerator=abs(apd1[i,j]-apd2[i,j])
                           denominator=(apd1[i,j]+apd2[i,j])/2                          
                           ap_coeff[i,j]=numerator/denominator
              
                       else: 
                           start1=np.argmin(imgff[0:peak1[i,j],i,j])
                           end1=peak1[i,j]
                           dvdtmax=np.argmax(np.diff(imgff[start1:
                                                           end1,i,j])) + start1
                           dvdtmax2=np.argmax(np.diff(
                               imgff[midP_idx[i,j].astype(int):
                                     peak2[i,j],i,j])) + midP_idx[i,j]
                           apd2[i,j]=(1000/fps)*(second_min_idx[i,j]-dvdtmax2)
                           mid_amp=np.round(midP[i,j],2)
                           mid_amp3=np.round(second_min[i,j],2)
             
                           win1=1000
                           min_inx=0
                           
                           for k in range(peak1[i,j], midP_idx[i,j].
                                          astype(int)):
                               if imgff[k,i,j] > (mid_amp and imgff[k,i,j] < 
                                                  mid_amp3+0.1): 
                                   win=abs(mid_amp3-imgff[k,i,j])
                                   if win<win1:
                                       win1=win
                                       min_inx=k
                          
                           apd1[i,j]=(1000/fps)*(min_inx-dvdtmax)
                           numerator=abs(apd1[i,j]-apd2[i,j])
                           denominator=(apd1[i,j]+apd2[i,j])/2                          
                           ap_coeff[i,j]=numerator/denominator
                           
           # duration calculation second beat 
           ap_coeff[ap_coeff== -inf] = 0
           #
           plt.figure('Trace 1 Durations')
           self.imaging_mapping(apd1, li1, li2, transp)
           plt.figure('Trace 2 Durations')
           self.imaging_mapping(apd2, li1, li2, transp)
           plt.figure('Alternan Coefficient')
           self.imaging_mapping(ap_coeff, li1, li2, transp)
           
           #making a folder if there isn't a "Saved Data Maps" folder
           if not os.path.exists("Saved Data Maps\\" + file_id):
               os.makedirs("Saved Data Maps\\" + file_id)
            
           #saving the data file           
           savetxt('Saved Data Maps\\' + file_id + '\\dynamic_AAC.csv', 
                   ap_coeff, delimiter=',')
           
        if image_type == 1:
            imgff=copy.deepcopy(img)
            imgff=normalize(imgff,start_ind,end_ind)
            aa=np.shape(imgff)
            aa=np.array(aa)
            midP, midP_idx, peak1, peak2 =self.midpoint(imgff,peak_coeff)
            min_inx=0
            ca_alt_coeff=np.zeros((aa[1],aa[2])) 
             
            for i in range(aa[1]):
                for j in range(aa[2]):
                    if peak2[i,j]>peak1[i,j] and peak2[i,j]>0:
                        min_point=np.min(imgff[0:peak1[i,j],i,j])
                        first=imgff[peak1[i,j],i,j]
                        second=imgff[peak2[i,j],i,j]
                        first_height=first-min_point
                        second_height=second-midP[i,j]
           
                        if first_height > second_height:
                           ca_alt_coeff[i,j]=1-(second_height/first_height)
                        else:
                            ca_alt_coeff[i,j]=1-(first_height/second_height)

            ca_alt_coeff[ca_alt_coeff == -inf] = 0
            self.imaging_mapping(ca_alt_coeff, li1, li2, transp)
            
            #making a folder if there isn't a "Saved Data Maps" folder
            if not os.path.exists("Saved Data Maps\\" + file_id):
               os.makedirs("Saved Data Maps\\" + file_id)
               
            #saving the data file
            savetxt('Saved Data Maps\\' + file_id + '\\dynamic_CAC.csv', 
                    ca_alt_coeff, delimiter=',')
       
        
    def post_analysis(self, li1, li2, file_id, file_loaded):          
        #assigning the working directory to a variable
        directory = os.getcwd()
        
        #adding the Saved Data Maps to the working directory path
        post_mapping_analysis_folder = str(directory) + '\Saved Data Maps\\' + file_id
        
        #checking if the saved data maps folder exists, if it doesn't the snr 
        #file also doesn't
        if not os.path.exists(post_mapping_analysis_folder):
            snr_file = 0
        else:
            #searching the Saved Data Maps folder for files named 'snr'
            for root, dirs, files in os.walk(post_mapping_analysis_folder):
                if 'snr.csv' in files:
                    snr_file = 1
                else:
                    snr_file = 0 
        
        #opening the Saved Data Maps file so the user can select a document
        if snr_file == 1:
            file_path = post_mapping_analysis_folder + '\\snr.csv'
            data_ori = loadtxt(file_path, delimiter=',')
            data_ori[np.isnan(data_ori)] = 0
            
            regional_maps_folder = str(directory) + '\\Saved Region Maps\\' + file_id
            
            #checking if a region folder exists, if not make one
            if not os.path.exists(regional_maps_folder):
                os.makedirs(regional_maps_folder)
            
            #checking if a region file has been created
            region_files = os.path.isfile(regional_maps_folder + '\\region_1.csv')
            
            #if there are any saved region files in the current file_id folder do:
            if region_files == True:
                region_file_list = os.listdir(regional_maps_folder)
                region_file_length = len(region_file_list)
                
                window = tkinter.Tk()
                window.wm_withdraw()
                message = ("""\nDo you want to use the """ + str(region_file_length) + " saved region selection(s)?")
                use_region = messagebox.askquestion(title="Reuse Region Selection", 
                                                    message=message)
                
                if use_region == 'no':
                    #creating a variable so we can save the inputed number of regions
                    region_number = tkinter.Tk()
                    #closing the weird tkinter window that pops up
                    region_number.wm_withdraw()
                    
                    #creating a dialogue box so the user can select the number of regions
                    region_value = tkinter.simpledialog.askstring(
                        'Region Number', 
                        """\nList the number of regions do you want to draw and select 'OK'.""", 
                        parent = region_number)
                    
                    #getting the region number and converting it from string to integers
                    x = int(region_value)

                    for j in range (x):
                        mask_ar=[]
                        img_mod=[]
                        tt=[]
                        dd=[]
                        duplicate=copy.deepcopy(data_ori)
                        dd=draw_poly(data_ori,li1,li2)
                        tt=dd.astype(int)
                        img_mod = cv2.fillPoly(duplicate, pts= [tt], color = (-10000, 0, 0))
                        mask_ar=np.argwhere(img_mod<-2000)
                       
                        savetxt('Saved Region Maps\\' + file_id + '\\region_'+str(j+1)+'.csv', 
                                mask_ar, delimiter=',')
                        
                #if the user wants to use the saved regions, let them decide how many
                #and which ones
                else:
                    #creating a variable so we can save the desired number of regions
                    region_number = tkinter.Tk()
                    #closing the weird tkinter window that pops up
                    region_number.wm_withdraw()
                    
                    #creating a dialogue box so the user can select the number of regions
                    region_value = tkinter.simpledialog.askstring(
                        'Region Number', 
                        """\nList which of the """ + str(region_file_length) + 
                        """ saved regions you want to use and select 'OK'.
                        \nInput example: 1,3,4""", 
                        parent = region_number)
                    
                    #x = int(region_value)
                    region_value_string = region_value.split(',')
                    x = [int(value) for value in region_value_string]
                    
            #if there are no saved regions, prompt the user to create them
            else:
                #creating a variable so we can save the inputed number of regions
                region_number = tkinter.Tk()
                #closing the weird tkinter window that pops up
                region_number.wm_withdraw()
                
                #creating a dialogue box so the user can select the number of regions
                region_value = tkinter.simpledialog.askstring(
                    'Region Number', 
                    """\nList the number of regions do you want to draw and select 'OK'.""", 
                    parent = region_number)
                
                #getting the region number and converting it from string to integers
                x = int(region_value)
                
                for j in range (x):
                    mask_ar=[]
                    img_mod=[]
                    tt=[]
                    dd=[]
                    duplicate=copy.deepcopy(data_ori)
                    dd=draw_poly(data_ori,li1,li2)
                    tt=dd.astype(int)
                    img_mod = cv2.fillPoly(duplicate, pts= [tt], color = (-10000, 0, 0))
                    mask_ar=np.argwhere(img_mod<-2000)
                    
                    savetxt('Saved Region Maps\\' + file_id +'\\region_'+str(j+1)+'.csv', 
                            mask_ar, delimiter=',')
                    
            #opening the file path that contains the files the user will want analyzed
            data_file_path = QFileDialog.getOpenFileName(
                None, "Open File", post_mapping_analysis_folder)
            
            #grabbing the file path, the second object [0] is just the file type
            #and not a part of the file path string
            data_ori= loadtxt(data_file_path[0], delimiter=',')
            
            selected_file_info = str(data_file_path)
            #splitting the file info into an array by the ,
            selected_file_info_obj = selected_file_info.split(',')
            #getting the selected file path
            selected_file_path = selected_file_info_obj[0]
            #splitting the file path into an array by the /
            selected_file_path_obj = selected_file_path.split('/')
            #getting the length of this object
            length = len(selected_file_path_obj)
            #grabbing the last object of the array, should be the file name the 
            #user selected
            full_file_name = selected_file_path_obj[length-1]
            
            file_name_not_type = full_file_name.replace('.csv', '')
            file_name = file_name_not_type.replace('"', '')
            
            loaded_file_path = file_loaded.split('.')
            loaded_file = loaded_file_path[0]
            
            header = loaded_file + '_' + file_name + "_region,mean,median,sd,n"
            
            if not os.path.exists("ROI Analysis\\" + file_id):
                os.makedirs("ROI Analysis\\" + file_id)

            with open("ROI Analysis\\" + file_id + '\\all_results.csv', 'a') as file:
                file.write(header + "\n")
                file.close
                file=open("ROI Analysis\\" + file_id + '\\all_results.csv','a')
                
                #creating a variable so we can save the inputed file name
                filename = tkinter.Tk()
                #closing the weird tkinter window that pops up
                filename.wm_withdraw()
                
                #creating a dialogue box so the user can create a file name
                file_appendage = tkinter.simpledialog.askstring(
                    'File Name', 
                    """\nType would you like to append to the file name and press enter. 
                       \nIf no name is desired, press enter.""", parent = filename)
               
                #if the user enters a file name append it to the file
                if file_appendage:
                    path_appendage = file_appendage
                #if the user does not enter a file name don't append any value
                else:
                    path_appendage = ''
                
                if not os.path.exists("ROI Analysis\\" + file_id + "\\" + file_name):
                    os.makedirs("ROI Analysis\\" + file_id + "\\" + file_name)
                
                #if the user wants specific regions, use their input to 
                #indicate what regions to use
                if type(x) == list:
                    x_length = len(x)
                    for i in range(x_length):
                        data=[]
                        mask_area=loadtxt('Saved Region Maps\\' + file_id +
                                          '\\region_'+str(x[i])+'.csv',delimiter=',')
                        mask_area=mask_area.astype(int)
                        data=data_ori[mask_area[:,0],mask_area[:,1]]
                       
                        savetxt('ROI Analysis\\' + file_id + '\\' +
                                file_name + "\\" + file_name + '_R' + str(x[i]) + 
                                '_' + path_appendage + 
                                '.csv', data, delimiter=',')
                        mean=round(np.mean(data),1)
                        median=round(np.median(data),1)
                        stdv=round(np.std(data),1)
                        n=np.size(data)
                        print(mean,median,stdv,n)
                        savetxt(file, np.c_[x[i],mean,median,stdv,n],fmt='%.1f', delimiter=',')
                #if the user doesn't want specific regions, continue with how many they want
                else:
                    for i in range(x):
                        data=[]
                        mask_area=loadtxt('Saved Region Maps\\' + file_id +
                                          '\\region_'+str(i+1)+'.csv',delimiter=',')
                        mask_area=mask_area.astype(int)
                        data=data_ori[mask_area[:,0],mask_area[:,1]]
                       
                        savetxt('ROI Analysis\\' + file_id + '\\' +
                                file_name + "\\" + file_name + '_R' + str(i+1) + 
                                '_' + path_appendage + 
                                '.csv', data, delimiter=',')
                        mean=round(np.mean(data),1)
                        median=round(np.median(data),1)
                        stdv=round(np.std(data),1)
                        n=np.size(data)
                        print(mean,median,stdv,n)
                        savetxt(file, np.c_[i+1,mean,median,stdv,n],fmt='%.1f', delimiter=',')
            file.close()
            
        else:
            window = tkinter.Tk()
            window.wm_withdraw()
            message = ("""\nAn SNR file is needed to use Region of Interest Analysis.
                  \n1. Select SNR under the 'Analysis' section and generate a map.
                  \n2. Then reselect Post Mapping Analysis.""")
            messagebox.showinfo(title="Warning", message=message)
        
    def S1_S2(self, fps, img, li1, li2, transp, start_ind, end_ind, 
                    start_ind2, end_ind2, interp_selection, perc_apd, 
                    image_type, peak_coeff, threshold, file_id):
      
        apd1= self.apd_analysis(fps, img, start_ind, end_ind, 
                                interp_selection, perc_apd, file_id)[0]
        
        imgf=copy.deepcopy(img)

        imgf=normalize(imgf,start_ind2,end_ind2)
        imgf[np.isnan(imgf)] = 0 
        
        if interp_selection > 0: 
            interp_selection=interp_selection+1
            imgf_2=self.interpol(imgf,interp_selection)
        else:
            interp_selection=interp_selection+1
            imgf_2=imgf
            
        aa=np.shape(imgf_2)
        aa=np.array(aa)
        
        imgff=imgf_2
        peak1=np.zeros((aa[1],aa[2]))
        peak2=np.zeros((aa[1],aa[2]))
        
        for i in range(aa[1]):
           for j in range(aa[2]): 
               hh, _ = find_peaks(imgff[:,i,j], height=threshold,width=peak_coeff)
               hh=hh.astype(int)
               hh_size=np.shape(hh)
               hh2=int(hh_size[0])
               if hh2 > 1 and hh2 <3:
                   peak1[i,j]=hh[0]
                   peak2[i,j]=hh[1]
           
               if hh2 > 2:   
                   if hh[1]-hh[0] < 40:  
                       peak1[i,j]=hh[0]
                       peak2[i,j]=hh[2]
                   else: 
                       peak1[i,j]=hh[0]
                       peak2[i,j]=hh[1]   
         
        peak1=peak1.astype(int)
        peak2=peak2.astype(int)
        min_inx=0
        midP=np.zeros((aa[1],aa[2]))
        midP_idx=np.zeros((aa[1],aa[2]))
        apd2=np.zeros((aa[1],aa[2]))
  
        for i in range(aa[1]):
           for j in range(aa[2]):
                if peak2[i,j]>peak1[i,j] and peak2[i,j]>0:
                    midP[i,j]=np.amin(imgff[peak1[i,j]:peak2[i,j],i,j])
                    midP_idx[i,j]=np.argmin((imgff[peak1[i,j]:peak2[i,j],i,j])+peak1[i,j])
                    dvdtmax=np.argmax(np.diff(imgff[midP_idx[i,j].astype(int):peak2[i,j],i,j]))+midP_idx[i,j]
                    win1=1000
                    min_inx=0
                    for k in range(peak2[i,j], (aa[0]*0.95).astype(int)):
                        
                        if imgff[k,i,j] > ((1-perc_apd) - 0.05) and imgff[k,i,j] < ((1-perc_apd) + 0.05):
                            win=abs((1-perc_apd)-imgff[k,i,j])                         

                            if win<win1:
                                win1=win
                                min_inx=k
                    apd2[i,j]=min_inx-dvdtmax 
                        
        apd2=(1000/fps)/interp_selection*apd2      
        diff=apd1-apd2   
        plt.figure('S1 Duration')
        self.imaging_mapping(apd1, li1, li2, transp)
        
        plt.figure('S2 Duration')
        self.imaging_mapping(apd2, li1, li2, transp)
        
        plt.figure('S1-S2 Duration Difference')
        self.imaging_mapping(diff, li1, li2, transp)    
        
        
        amp0=np.zeros((aa[1],aa[2]))
        amp1=np.zeros((aa[1],aa[2]))
        amp_coeff=np.zeros((aa[1],aa[2]))
        peak0=np.zeros((aa[1],aa[2]))
        imgg=copy.deepcopy(img)
        imgf2=normalize(imgg,start_ind,end_ind)
        
        imgf2[np.isnan(imgf2)] = 0
        imgf2_2=imgf2
        
        if interp_selection > 0: 
            interp_selection=interp_selection+1
            imgf2_2=self.interpol(imgf2,interp_selection)
        else:
            interp_selection=interp_selection+1
            imgf2_2=imgf2                   
        
        for i in range(aa[1]): 
            for j in range(aa[2]): 
                hh0, _ = find_peaks(imgf2_2[:,i,j], height=threshold, 
                                    width=peak_coeff)
                hh0=hh0.astype(int)
                hh0_size=np.shape(hh0)
                hh1=int(hh0_size[0])
                if hh1 >= 1:
                    peak0[i,j]=hh0[0]
        
        peak0=peak0.astype(int)
        
        # amplitude calculation
        for i in range(aa[1]):
           for j in range(aa[2]): 
               amp0[i,j]=imgf2_2[peak0[i,j],i,j]
               amp1[i,j]=imgff[peak2[i,j],i,j]-imgff[midP_idx[i,j].astype(int),
                                                     i,j]
               amp_coeff[i,j]=(1-(amp1[i,j]/amp0[i,j]))*100
               
        amp_coeff[amp_coeff == -inf] = 0       
        
        plt.figure('Peak')
        self.imaging_mapping(peak0, li1, li2, transp)   
        plt.figure('Amplitude 1')
        self.imaging_mapping(amp0, li1, li2, transp)  
        plt.figure('Amplitude 2')
        self.imaging_mapping(amp1, li1, li2, transp)  
        plt.figure('Amplitude Coefficient')
        self.imaging_mapping(amp_coeff, li1, li2, transp)
        
        #making a folder if there isn't a "Saved Data Maps" folder
        if not os.path.exists("Saved Data Maps\\" + file_id):
           os.makedirs("Saved Data Maps\\" + file_id)
               
        #saving data file
        savetxt('Saved Data Maps\\' + file_id + '\\S2'+str(perc_apd*100)+'.csv', apd2, 
                delimiter=',')  
        savetxt('Saved Data Maps\\' + file_id + '\\S1S2_difference'+str(perc_apd*100)+'.csv', 
                diff, delimiter=',') 
        savetxt('Saved Data Maps\\' + file_id + '\\S1S2_amp_coeff'+str(perc_apd*100)+'.csv', 
                amp_coeff, delimiter=',') 