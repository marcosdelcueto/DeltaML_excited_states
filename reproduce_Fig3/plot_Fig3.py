#!/usr/bin/env python3
#################################################################################
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import stats
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from mpl_toolkits.axes_grid1 import make_axes_locatable

def main():
    # Define files with data
    db_file      = 'results_train.csv'
    db_file_test = 'results_test.csv'
    # Read data as dataframe
    df      = pd.read_csv(db_file,index_col=None)
    df_test = pd.read_csv(db_file_test,index_col=None)
    # Call function to do plots
    do_plots(df,df_test)

def do_plots(df,df_test):
    # Assign X, Y data from dataframes
    S1_ZINDO           = df['S1_ZINDO'].values.reshape(-1, 1)
    S1_TDDFT           = df['S1_TDDFT'].values.reshape(-1, 1)
    S1_ML              = df['S1_ML'].values.reshape(-1, 1)
    S1_ZINDO_test      = df_test['S1_ZINDO'].values.reshape(-1, 1)
    S1_TDDFT_test      = df_test['S1_TDDFT'].values.reshape(-1, 1)
    S1_ML_test         = df_test['S1_ML'].values.reshape(-1, 1)
    # Calculate data with linear regressions
    # train ZINDO-TDDFT
    linear_regressor = LinearRegression() # create object for the class
    linear_regressor.fit(S1_ZINDO, S1_TDDFT)            # perform linear regression
    S1_lin = linear_regressor.predict(S1_ZINDO)   # make predictions
    # test ZINDO-TDDFT
    linear_regressor_test = LinearRegression()  # create object for the class
    linear_regressor_test.fit(S1_ZINDO_test, S1_TDDFT_test)  # perform linear regression
    S1_lin_test = linear_regressor_test.predict(S1_ZINDO_test)  # make predictions
    # reshape data
    S1_ZINDO           = S1_ZINDO.reshape(1,-1)[0]
    S1_ZINDO_test      = S1_ZINDO_test.reshape(1,-1)[0]
    S1_ML        = S1_ML.reshape(1,-1)[0]
    S1_ML_test   = S1_ML_test.reshape(1,-1)[0]
    S1_TDDFT           = S1_TDDFT.reshape(1,-1)[0]
    S1_TDDFT_test      = S1_TDDFT_test.reshape(1,-1)[0]
    # calculate r and rmse for train/test for each plot
    r,_         = pearsonr(S1_ZINDO, S1_TDDFT)
    r_test,_    = pearsonr(S1_ZINDO_test, S1_TDDFT_test)
    r_ML,_      = pearsonr(S1_ML, S1_TDDFT)
    r_ML_test,_ = pearsonr(S1_ML_test, S1_TDDFT_test)
    rmse        = math.sqrt(mean_squared_error(S1_lin,S1_TDDFT))
    rmse_test   = math.sqrt(mean_squared_error(S1_lin_test,S1_TDDFT_test))
    rmse_ML     = math.sqrt(mean_squared_error(S1_ML,S1_TDDFT))
    rmse_ML_test= math.sqrt(mean_squared_error(S1_ML_test,S1_TDDFT_test))
    # Prepare layout
    fig, axs = plt.subplots(ncols=2, nrows=1)
    plt.subplots_adjust(wspace=-0.20, hspace=-0.20,left=0.05, bottom=0.05, right=0.95, top=0.95)
    # Contour plots parameters
    # We divide each dimension in a number of bins, then count molecules in each bin
    levelsf = np.arange(0, 150, 0.1)   # range for color scale
    levels  = np.linspace(10, 110, 6)  # range for lines
    bins=(50,50)                       # how many bins per dimension
    cmap=cm.BuPu                       # colormap name

    # Panel A
    # do 2d binning
    H, xedges, yedges = np.histogram2d(S1_ZINDO, S1_TDDFT,bins=bins )
    xmesh, ymesh = np.meshgrid(xedges[:-1], yedges[:-1])
    # plot colormap with 'contourf', and contour lines with 'contour'
    cset1 = axs[0].contourf(xmesh,ymesh,H.T,levels=levelsf, cmap=cmap,extend='max',alpha=1.0,zorder=1)
    cset2 = axs[0].contour(xmesh,ymesh,H.T,levels=levels,zorder=2,linestyles='solid',colors='k',linewidths=0.5)
    # plot labels for contour lines
    axs[0].clabel(cset2, cset2.levels, inline=False, fontsize=0,inline_spacing=-3)
    #axs[0].scatter(S1_ZINDO_test, S1_TDDFT_test, alpha=0.75,edgecolors='C1',color='C1', linewidth=0.5,s=10) # plot test data
    # plot colorbar
    axs[0].set_xlabel(r"$E^{S1}_{ZINDO}\, (eV)$",  size=20, labelpad=12)
    axs[0].set_ylabel(r"$E^{S1}_{TDDFT}\, (eV)$", size=20, labelpad=12)
    axs[0].set_xlim(2.0,5.50)
    axs[0].set_ylim(2.0,5.50)
    axs[0].plot(np.arange(0, 8, 0.1), np.arange(0, 8, 0.1), color="k", ls="--")
    axs[0].text(0.05,1.08,'A)',fontsize=24,weight='bold',ha='right', va='top',transform=axs[0].transAxes)

    # Panel B
    # do 2d binning
    H, xedges, yedges = np.histogram2d(S1_ML, S1_TDDFT,bins=bins )
    max_H=0
    for i in range(len(H)):
        if max(H[i]) > max_H: max_H = max(H[i])
    xmesh, ymesh = np.meshgrid(xedges[:-1], yedges[:-1])
    # plot colormap with 'contourf', and contour lines with 'contour'
    cset1 = axs[1].contourf(xmesh,ymesh,H.T,levels=levelsf, cmap=cmap, extend='max',alpha=1.0,zorder=1)
    cset2 = axs[1].contour(xmesh,ymesh,H.T,levels=levels,zorder=2,linestyles='solid',colors='k',linewidths=0.5)
    # plot labels for contour lines
    axs[1].clabel(cset2, cset2.levels, inline=False, fontsize=0,inline_spacing=-3)
    #axs[1].scatter(S1_ML_test, S1_TDDFT_test, alpha=0.75,edgecolors='C1',color='C1', linewidth=0.5,s=10) # plot test data
    # plot colorbar
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar=fig.colorbar(cset1,cax=cax,orientation='vertical')
    cbar.ax.tick_params(labelsize=15)
    cbar.ax.set_ylabel('Number of points', rotation=90,size=20)
    axs[1].set_xlabel(r"$E^{S1}_{ML}\, (eV)$",  size=20, labelpad=12)
    axs[1].set_xlim(2.0,5.50)
    axs[1].set_ylim(2.0,5.50)
    axs[1].plot(np.arange(0, 8, 0.1), np.arange(0, 8, 0.1), color="k", ls="--")
    axs[1].text(0.05,1.08,'B)',fontsize=24,weight='bold',ha='right', va='top',transform=axs[1].transAxes)

    # Font sizes ticks
    axs[0].set_xticklabels(np.arange(2.0, 6.0, 0.5),fontsize=15)
    axs[0].set_yticklabels(np.arange(2.0, 6.0, 0.5),fontsize=15)
    axs[1].set_xticklabels(np.arange(2.0, 6.0, 0.5),fontsize=15)
    axs[1].set_yticklabels([])

    # Plot to file
    file_name='Fig3.png'
    figure=plt.gcf()
    figure.set_size_inches(16,8)
    plt.tight_layout()
    plt.savefig(file_name,dpi=300)

if __name__ == '__main__':
    main()
