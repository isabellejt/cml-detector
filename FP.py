import streamlit as st
from matplotlib import pyplot as plt
import numpy as np
from skimage import filters
from skimage import morphology
from PIL import Image
import scipy.ndimage as ndi
from skimage.measure import label, regionprops, regionprops_table
import pandas as pd


def find_histogram(image):
    hist=ndi.histogram(image,min=0,max=255,bins=256)
    return hist

def select_filter(filter,image):
    if filter=='isodata':
        thresh=filters.threshold_isodata(image)
    elif filter=='li':
        thresh=filters.threshold_li(image)
    elif filter=='mean':
        thresh=filters.threshold_mean(image)
    elif filter=='minimum':
        thresh=filters.threshold_minimum(image)
    elif filter=='otsu':
        thresh=filters.threshold_otsu(image)
    elif filter=='triangle':
        thresh=filters.threshold_triangle(image)
    elif filter=='yen':
        thresh=filters.threshold_yen(image)
    else:
        return False
    return thresh

def main():
    st.title("Final Project of Medical Image Processing")
    menu=st.sidebar.selectbox("What do you want to do?",['Main Menu','The Project', 'About Me'])
    
    area=[]

    if menu=='Main Menu':
        st.subheader('This project will detect CML from Microscopic Image')

    elif menu=='The Project':
        raw_image=Image.open("cml-sample-image.jpg")
        gray_image=raw_image.convert('L')
        gray_image=np.array(gray_image)
        median_filtered=filters.median(gray_image)
        if st.button("Raw Image"):
            hist=find_histogram(raw_image)
            st.image(raw_image)
            st.info(f"""image format: {raw_image.format}; description: {raw_image.format_description}
                    ;size: {raw_image.size}""")
            st.line_chart(hist)

        if st.button("Increase contrast"):
            hist=find_histogram(median_filtered)
            st.line_chart(hist)
            fig=plt.figure()
            plt.imshow(median_filtered,cmap='gray')
            plt.axis(False)
            st.pyplot(fig)

            st.write("Check for the best threshold")
            fig_thresh, axes_thresh=filters.try_all_threshold(median_filtered,figsize=(20,15)) 
            st.pyplot(fig_thresh)

        filter_choice=st.selectbox("Select the best filter",
                                   ['isodata','li','mean','minimum','otsu','triangle','yen'],
                                   index=None)                
        if filter_choice is not None:
            thresh=select_filter(filter_choice,median_filtered)
            binary_image=median_filtered<thresh
            size_erosion=st.slider("erosion value",min_value=1,max_value=10,value=2)
            size_dilation=st.slider("dilation value",min_value=1,max_value=10,value=7)
            iamge_cleaned=ndi.grey_erosion(binary_image, size=size_erosion)
            image_cleaned_=ndi.grey_dilation(iamge_cleaned, size=size_dilation)
            image_cleaned1 = morphology.remove_small_objects(image_cleaned_, min_size=500)
            if st.button("Threshold"):                
                fig, ax = plt.subplots(ncols=3, figsize=(20, 5))
                ax[0].imshow(binary_image, cmap="gray")
                ax[0].set_title('After Thresholding')
                ax[1].imshow(image_cleaned_, cmap="gray")
                ax[1].set_title('After Dilation and Erosion')
                ax[2].imshow(image_cleaned1, cmap="gray")
                ax[2].set_title('Removing Small Objects')
                for ax_ in ax:
                    ax_.axis(False)
                st.pyplot(fig)

            labels, nlabels = ndi.label(image_cleaned1)
            labels_for_display = np.where(labels > 0, labels, np.nan)
            label_img = label(image_cleaned1)
            regions = regionprops(label_img)
            if st.button('CML Detector'):
                fig=plt.figure()
                plt.title("CML detected")
                plt.imshow(image_cleaned1,cmap='gray')
                plt.axis(False)
                plt.imshow(labels_for_display, cmap='rainbow')
                st.pyplot(fig)
                st.write(f"There are {nlabels} CML cells in this image.")
                
                # fig,axes= plt.subplots(nrows=1,ncols=2,figsize=(25,8))
                # fig.suptitle("CML detected",fontsize=30)
                # axes[0].imshow(raw_image)
                # axes[0].axis(False)
                # axes[1].imshow(image_cleaned1,cmap='gray')
                # axes[1].imshow(labels_for_display, cmap='rainbow')
                # axes[1].axis(False)
                # st.pyplot(fig)

                fig, axes = plt.subplots(nrows=1, ncols=nlabels, figsize=(20, 5))
                for ii, obj_indices in enumerate(ndi.find_objects(labels)):
                    cell = image_cleaned1[obj_indices]
                    area.append(np.count_nonzero(cell))
                    axes[ii].imshow(cell, cmap="gray")
                    axes[ii].axis(False)
                    axes[ii].set_title("Object #{}\nSize: {}\n Area: {}".format(ii, cell.shape,area[ii]))
                st.pyplot(fig)

                fig, ax = plt.subplots()
                ax.imshow(image_cleaned1, cmap='gray')
                for props in regions:
                    y0, x0 = props.centroid
                    orientation = props.orientation
                    x1 = x0 + np.cos(orientation) * 0.5 * props.minor_axis_length
                    y1 = y0 - np.sin(orientation) * 0.5 * props.minor_axis_length
                    x2 = x0 - np.sin(orientation) * 0.5 * props.major_axis_length
                    y2 = y0 - np.cos(orientation) * 0.5 * props.major_axis_length

                    ax.plot((x0, x1), (y0, y1), '-b', linewidth=1)
                    ax.plot((x0, x2), (y0, y2), '-b', linewidth=1)
                    ax.plot(x0, y0, '.r', markersize=3)

                    minr, minc, maxr, maxc = props.bbox
                    bx = (minc, maxc, maxc, minc, minc)
                    by = (minr, minr, maxr, maxr, minr)
                    ax.plot(bx, by, '-g', linewidth=1)
                plt.axis(False)
                st.pyplot(fig) 

                properties=regionprops_table(label_img,properties=
                                             ['label','centroid','orientation',
                                              'major_axis_length','minor_axis_length','perimeter'])
                data=pd.DataFrame(properties)
                data['Area']=area
                st.dataframe(data)
                data.to_csv('tempdir/CML',sep=' ',index=False)
                st.success('Data saved as CSV')

    elif menu=='About Me':
        st.write('Isabelle Jessica Tjitalaksana')
        st.write('Biomedical Engineering 2021')
        st.write('NRP 5023211022')
    

if __name__=="__main__":
    main()