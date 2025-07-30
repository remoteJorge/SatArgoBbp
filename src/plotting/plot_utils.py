import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.style.use(['seaborn-v0_8-whitegrid'])

def get_depth_plot_params(setup):
    depth_value = int(setup.depth)
    region = setup.region
    experiment = setup.experiment

    # Depth ticks and range
    if depth_value == 50:
        depth_range = np.arange(1, 52, 2)
        yticks = (-1, -11, -21, -31, -41, -51)
    elif depth_value == 250:
        depth_range = np.arange(1, 253, 2)
        yticks = (-1, -51, -101, -151, -201, -250)
    else:
        raise ValueError(f"Unknown depth: {depth_value}")

    # Region color
    color = 'g' if region == 'NA' else '#1f77b4'

    # Experiment linestyle
    linestyle = '--' if "S3" in experiment else '-'

    return depth_range, yticks, color, linestyle

def plot_r2_depth_profile(depth_metrics_df, setup, exports):
    """
    Plots R² vs. depth profile and optionally saves it.

    Args:
        r2_depth (array-like): R² values per depth.
        setup (ExperimentSetup): Contains region, depth, experiment info.
        exports (ExportConfig): Controls whether to export the figure.
    """
    r2_depth = depth_metrics_df.loc["R2"].values
    depth_range, yticks, color, linestyle = get_depth_plot_params(setup)

    # Plot
    plt.figure(dpi=300)
    plt.plot(r2_depth, -depth_range, color=color, alpha=0.7, linestyle=linestyle)
    plt.yticks(yticks, fontsize=14)
    plt.ylabel('Depth (m)', fontsize=16)
    plt.xlabel("R²", fontsize=14)
    plt.xticks(fontsize=14)

    # Export if enabled
    if exports.export_fig_depth:
        fig_dir = f"../results/{setup.region}/{setup.depth}/{setup.experiment}/Model/Figures"
        os.makedirs(fig_dir, exist_ok=True)
        filename = f"{setup.region}_{setup.depth}_{setup.experiment}_R2_depth.png"
        plt.savefig(os.path.join(fig_dir, filename), bbox_inches='tight')

    plt.show()

# Vars for Feature Importances colors
# Define feature groups
feature_groups = {
    "Spatio_temporal": ["doy", "lon", "lat"],
    "SLA": ["SLA"],
    "MLD": ["MLD"],
    "PAR_GlbClr": ["PAR", "GlbClr_412", "GlbClr_443", "GlbClr_490", "GlbClr_555", "GlbClr_670"],
    "S3": ["S3b1_400", "S3b2_412.5", "S3b3_442.5", "S3b4_490", "S3b5_510", "S3b6_560", "S3b7_620",
           "S3b8_665", "S3b9_673.75", "S3b10_681.25", "S3b11_708.75", "S3b12_753.75"],
    "Dens": ["Dens_pc1", "Dens_pc2", "Dens_pc3", "Dens_pc4", "Dens_pc5"],
    "Temp": ["Temp_pc1", "Temp_pc2", "Temp_pc3", "Temp_pc4", "Temp_pc5"],
    "Sal": ["Sal_pc1", "Sal_pc2", "Sal_pc3", "Sal_pc4", "Sal_pc5"],
    "Spiciness": ["Spi_pc1", "Spi_pc2", "Spi_pc3", "Spi_pc4", "Spi_pc5"]
}

# Assign colors to each feature group (neutral/soft colors)
fi_colors = {
    "Spatio_temporal": "#A6CEE3",
    "SLA": "#1F78B4",
    "MLD": "#9370DB",
    "PAR_GlbClr": "#33A02C",
    "S3": "#33A02C",
    "Dens": "#FB9A99",
    "Temp": "#E31A1C",
    "Sal": "#FDBF6F",
    "Spiciness": "#ab9bb3",
    "IOPS": "#D3D3D3"  
}

def plot_feature_importance(model, x_train, setup, export_config):
    """
    Plots and optionally saves the feature importance as a vertical bar chart.

    Args:
        model: Trained model (e.g., RandomForestRegressor)
        x_train: Training DataFrame (used for feature names)
        setup: ExperimentSetup object
        export_config: ExportConfig object to control saving
    """
    # Compute feature importance
    fi_values = model.feature_importances_
    df_fi = pd.Series(fi_values, index=x_train.columns)

    # Assign colors based on feature groups
    feature_colors = [
        next(
            (fi_colors[group] for group in feature_groups if feature in feature_groups[group]),
            fi_colors["IOPS"]  # default fallback color
        )
        for feature in df_fi.index
    ]

    # Plot
    plt.style.use(['seaborn-v0_8-whitegrid'])
    plt.figure(figsize=(4, 10))
    df_fi.plot(kind='barh', fontsize=12, color=feature_colors)
    plt.title(f"{setup.region} {setup.depth}m {setup.experiment}", fontsize=14)
    plt.xlabel("Importance")
    plt.xticks(fontsize=10)
    plt.ylabel("")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    # Save if required
    if export_config.export_fig_importance:
        folder = f"../results/{setup.region}/{setup.depth}/{setup.experiment}/Model/Figures"
        os.makedirs(folder, exist_ok=True)
        fig_path = f"{folder}/{setup.region}_{setup.depth}_{setup.experiment}_Importance.pdf"
        plt.savefig(fig_path)

    plt.show()
    
def plot_bbp_profiles(y_test_bbp, y_pred_bbp, setup, export_config):
    """
    Plot measured and predicted bbp profiles side by side.

    Args:
        y_test_bbp (np.ndarray): Measured bbp values.
        y_pred_bbp (np.ndarray): Predicted bbp values.
        setup (ExperimentSetup): Experiment setup object.
        export_config (ExportConfig): Export configuration.
    """

    # Y-ticks and labels
    if setup.depth == '50':
        yticks = [0, 5, 10, 15, 20, 25]
        yticklabels = ['0', '-11', '-21', '-31', '-41', '-51']
    elif setup.depth == '250':
        yticks = [0, 25, 50, 75, 100, 125]
        yticklabels = ['0', '-50', '-100', '-150', '-200', '-250']
    else:
        raise ValueError(f"Unsupported depth: {setup.depth}")

    # Colorscale
    norm = norm = colors.LogNorm(vmin=np.min(y_test_bbp), vmax=np.max(y_test_bbp))
    
    # Plot
    plt.style.use(['bmh'])
    fig, ax = plt.subplots(1, 2, figsize=(12, 3), sharey='all', constrained_layout=True)

    im1 = ax[0].imshow(y_test_bbp.T, cmap='viridis', aspect='auto', norm=norm)
    ax[0].set_title('Measured $b_{bp}$', fontsize=14)
    ax[0].set_yticks(yticks)
    ax[0].set_yticklabels(yticklabels, fontsize=10)
    ax[0].set_ylabel('Depth (m)', fontsize=12)
    ax[0].set_xlabel('Profiles', fontsize=12)
    ax[0].xaxis.set_tick_params(labelsize=10)

    im2 = ax[1].imshow(y_pred_bbp.T, cmap='viridis', aspect='auto', norm=norm)
    ax[1].set_title('Predicted $b_{bp}$', fontsize=14)
    ax[1].set_xlabel('Profiles', fontsize=12)
    ax[1].xaxis.set_tick_params(labelsize=10)

    vmin = np.min(y_test_bbp.values)
    vmax = np.max(y_test_bbp.values)
    vcenter = np.sqrt(vmin * vmax)

    cbar = plt.colorbar(
	 im2,
	 ax=ax[1],
	 format='%.4f',
	 ticks=[vmin, vcenter, vmax])
	 
    cbar.set_label('$b_{bp} (m^{-1})$', fontsize=14)
    cbar.ax.tick_params(labelsize=10)

    # Export if requested
    if export_config.export_fig_profile:
        save_path = f"../results/{setup.region}/{setup.depth}/{setup.experiment}/Model/Figures"
        os.makedirs(save_path, exist_ok=True)
        fig_path = f"{save_path}/{setup.region}_{setup.depth}_{setup.experiment}_Profiles.png"
        plt.savefig(fig_path, dpi=300)

    plt.show()
    
    
def plot_relative_error(y_test_bbp, y_pred_bbp, setup, export_config):
    """
    Plots a heatmap of the relative error along with a vertical histogram by depth.

    Parameters:
    - relative_errors_bbp: 2D array of relative errors (profiles x depths)
    - setup: ExperimentSetup instance containing region, depth, experiment
    - export_config: ExportConfig instance with export_fig_profile flag
    """
    # Compute relative error
    relative_errors_bbp = 100 * (np.abs(y_test_bbp - y_pred_bbp) / y_test_bbp)
    relative_errors_bbp = np.clip(relative_errors_bbp, 0, 100)

    # Style and figure
    plt.style.use(['seaborn-v0_8-whitegrid'])
    fig, ax = plt.subplots(figsize=(7, 3.2))

    # Main heatmap
    im = ax.imshow(
        relative_errors_bbp.T,
        cmap='magma',
        vmin=0,
        vmax=100,
        aspect='auto',
        origin='lower'
    )
    ax.set_xlabel('Profiles')
    ax.xaxis.set_tick_params(labelsize=10)
    ax.set_title('Relative Error')
    ax.set_yticklabels([])

    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='3%', pad=0.15)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Relative Error (%)', rotation=90, labelpad=4, fontsize=10)
    cbar.ax.tick_params(labelsize=10)

    # Add vertical histogram
    hax = divider.append_axes('right', size='15%', pad=0.6, sharey=ax)
    depth_mean_re = np.mean(relative_errors_bbp, axis=0)
    depth_indices = np.arange(relative_errors_bbp.shape[1])
    hax.barh(depth_indices, depth_mean_re, color='gray', alpha=0.7)
    hax.set_ylim(ax.get_ylim())
    hax.invert_yaxis()
    hax.tick_params(labelleft=False)
    hax.set_ylabel('Mean Relative Error by depth (%)', rotation=90, labelpad=15, fontsize=10)
    hax.yaxis.set_label_position('right')
    hax.xaxis.set_tick_params(labelsize=10)

    plt.tight_layout()

    # Export if requested
    if export_config.export_fig_profile:
        save_path = f"../results/{setup.region}/{setup.depth}/{setup.experiment}/Model/Figures"
        os.makedirs(save_path, exist_ok=True)
        fig_path = f"{save_path}/{setup.region}_{setup.depth}_{setup.experiment}_Profiles_RE.png"
        plt.savefig(fig_path, dpi=300)

    plt.show()



def plot_validation_scatter(df_metrics, y_val_bbp, y_pred_val_bbp, setup, export_config):
    """
    Plots a scatter plot of measured vs. predicted values with depth as color and annotated metrics.

    Parameters:
    - df_metrics: DataFrame with validation metrics (must include R2, MAPD, MAE, RMSE rows)
    - vals: DataFrame or Series of measured values
    - preds: DataFrame or Series of predicted values
    - setup: ExperimentSetup instance
    - export_config: ExportConfig instance with export_fig_val flag
    """
    # Convert inputs to flat lists
    y_pred_val_bbp_list = list(y_pred_val_bbp.flat)
    y_val_bbp_list = list(y_val_bbp.to_numpy().flat)
    
    preds = pd.DataFrame (y_pred_val_bbp_list)
    vals = pd.DataFrame (y_val_bbp_list)

    # Get depth profile
    depth_range, _, _, _ = get_depth_plot_params(setup)
    np_depth_arr = np.tile(depth_range, 141)
    profun = pd.DataFrame(np_depth_arr)
    
    pieces = [vals, preds, profun]#, errordf[mask]]
    df_plot = pd.concat (pieces, axis=1)
    df_plot.columns = ['Measured', 'Predicted', 'Depths']#, 'Errors']

    
    # Set figure style
    plt.figure(figsize=(8,8))
    plt.style.use(['bmh'])
    sns.set_style(rc = {'axes.facecolor': 'white'})
    
    # Create joint plot (scatter only, no regression line)
    g = sns.jointplot(x="Measured", y="Predicted", data=df_plot, kind='reg',color='white',
	    line_kws={'linewidth': 0.4,'color': 'grey'},
	    marginal_kws={'bins': 20,'kde': False,'color':sns.color_palette()[0]})

    # Add colored scatter points
    pts = g.ax_joint.scatter(df_plot["Measured"], df_plot["Predicted"], c= -df_plot["Depths"], 
            edgecolors='grey',linewidth=0.2,cmap='viridis', s=50)

    # Remove regression line (already done by using kind='scatter')
    # Move colorbar
    cbar = plt.colorbar(pts, ax=g.ax_joint)
    cbar.ax.set_ylabel("Depth (m)", fontsize=12)

    # Set labels and title
    g.ax_joint.set_xlabel(r'Measured $log_{10}$ $(b_{bp} (m^{-1}))$', fontsize=12)
    # Define fixed y-axis limits
    y_limits = {
    "NA": {
    "50": (-3.3, -2.2),
    "250": (-3.6, -2.5),
    },
    "STG": {
    "50": (-3.43, -3.2),
    "250": (-3.8, -3.1),
    }
    }

    # Determine y-axis limits based on the region and depth
    fixed_ylim = y_limits.get(setup.region, {}).get(setup.depth, {})
    
    # Apply the y-axis limits if they exist
    if fixed_ylim:
        g.ax_joint.set_ylim(fixed_ylim)
        g.ax_joint.set_ylabel(r'Predicted $log_{10}$ $(b_{bp}(m^{-1}))$', fontsize=12)
        g.fig.suptitle(f"{setup.region} {setup.depth}m {setup.experiment}", fontsize=14)

    r2_val = float(df_metrics.loc["R2"].mean())
    mapd_val = float(df_metrics.loc["MAPD"].median())
    mae_val = float(df_metrics.loc["MAE"].mean())
    rmse_val = float(df_metrics.loc["RMSE"].mean())
    
    # Compute fixed text position based on axis limits
    x_min, x_max = g.ax_joint.get_xlim()
    y_min, y_max = g.ax_joint.get_ylim()
    
    x_pos = x_min + 0.05 * (x_max - x_min)  # 2% from left
    y_pos = y_max - 0.05 * (y_max - y_min)  # 10% from top

    # Add annotation with fixed relative position
    g.ax_joint.text(
        x_pos, y_pos,
        (
             rf"$R^2$: {r2_val:.2f}"+"\n"
             rf"MAPD: {mapd_val:.2f} "+"\n"
             rf"MAE: {mae_val:.2E} m$^{{-1}}$"+"\n"
             rf"RMSE: {rmse_val:.2E} m$^{{-1}}$"
    ),
    fontsize=11,
    verticalalignment='top',
    horizontalalignment='left',
    bbox=dict(facecolor='white', alpha=0.8, edgecolor='grey')
    )
    
    plt.tight_layout()
    
    # Save figure
    if getattr(export_config, "export_fig_val", False):
       save_path = f"../results/{setup.region}/{setup.depth}/{setup.experiment}/Validation/Figures"
       os.makedirs(save_path, exist_ok=True)
       fig_path = os.path.join(save_path, f"{setup.region}_{setup.depth}_{setup.experiment}_validation.pdf")
       plt.savefig(fig_path, dpi=300)
       plt.show()
       
       
def plot_bgc_sat_experiments(
    r2_depth, mae_depth, bias_depth, depth, export=True
):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import ScalarFormatter

    mld_na=-36 
    mld_stg=-50 

    # Define depth-related parameters
    if str(depth) == "50":
        depth_range = np.arange(1, 52, 2)
        yticks = (-1, -11, -21, -31, -41, -51)
    elif str(depth) == "250":
        depth_range = np.arange(1, 253, 2)
        yticks = (-1, -51, -101, -151, -201, -250)
    else:
        raise ValueError("Unsupported depth value. Use '50' or '250'.")

    # Formatter
    formatter = ScalarFormatter()
    formatter.set_scientific(True)
    formatter.set_powerlimits((-3, 2))

    fig, axes = plt.subplots(
        1, 6, figsize=(16, 5), dpi=300,
        gridspec_kw={'width_ratios': [1, 1, 0.5, 0.5, 0.5, 0.5]},
        sharey=False
    )

    colors = {'NA': 'g', 'STG': '#1f77b4'}

    # --- R² Plot ---
    ax = axes[0]
    ax.axhline(y=mld_stg, color=colors['STG'], linewidth=3, alpha=0.2, label='Mean STG MLD')
    ax.axhline(y=mld_na, color=colors['NA'], linewidth=3, alpha=0.2, label='Mean NA MLD')
    ax.plot(r2_depth["NA_S3OLCIBGC"], -depth_range, color=colors['NA'], alpha=0.7, linestyle="--", linewidth=0.9, label="NA S3OLCIBGC")
    ax.plot(r2_depth["NA_GCGOBGC"], -depth_range, color=colors['NA'], alpha=0.7, linestyle="-", linewidth=0.9, label="NA GCGOBGC")
    ax.plot(r2_depth["STG_S3OLCIBGC"], -depth_range, color=colors['STG'], alpha=0.7, linestyle="--", linewidth=0.9, label="STG S3OLCIBGC")
    ax.plot(r2_depth["STG_GCGOBGC"], -depth_range, color=colors['STG'], alpha=0.7, linestyle="-", linewidth=0.9, label="STG GCGOBGC")
    ax.set_xlabel("R\u00B2", fontsize=12)
    ax.set_ylabel("Depth (m)", fontsize=12)
    ax.set_yticks(yticks)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=8)

    # --- MAE Plot ---
    ax = axes[1]
    ax.axhline(y=mld_stg, color=colors['STG'], linewidth=3, alpha=0.2)
    ax.axhline(y=mld_na, color=colors['NA'], linewidth=3, alpha=0.2)
    ax.set_yticklabels([])
    ax.plot(mae_depth["NA_S3OLCIBGC"], -depth_range, color=colors['NA'], alpha=0.7, linestyle="--", linewidth=0.9)
    ax.plot(mae_depth["NA_GCGOBGC"], -depth_range, color=colors['NA'], alpha=0.7, linestyle="-", linewidth=0.9)
    ax.plot(mae_depth["STG_S3OLCIBGC"], -depth_range, color=colors['STG'], alpha=0.7, linestyle="--", linewidth=0.9)
    ax.plot(mae_depth["STG_GCGOBGC"], -depth_range, color=colors['STG'], alpha=0.7, linestyle="-", linewidth=0.9)
    ax.set_yticks(yticks)
    ax.set_xlabel("MAE $(m^{-1})$", fontsize=12)
    ax.tick_params(labelsize=10)
    axes[1].xaxis.set_major_formatter(formatter)

    # --- Bias bar plots ---
    for i, (region, dataset) in enumerate([
        ("NA", "S3OLCIBGC"), ("NA", "GCGOBGC"),
        ("STG", "S3OLCIBGC"), ("STG", "GCGOBGC")
    ]):
        ax = axes[i + 2]
        ax.barh(depth_range, bias_depth[f"{region}_{dataset}"], color=colors[region], alpha=0.6)
        ax.set_yticklabels([])
        ax.invert_yaxis()
        ax.axvline(x=0, color='darkgrey', linestyle='--', linewidth=1)
        ax.tick_params(labelsize=10)
        ax.set_title(dataset, fontsize=11)
        ax.xaxis.set_major_formatter(formatter)

    # Auto-scale bias plot x-limits by region
    for region, idxs in zip(["NA", "STG"], [(2, 3), (4, 5)]):
        bias_values = np.concatenate([bias_depth[f"{region}_S3OLCIBGC"], bias_depth[f"{region}_GCGOBGC"]])
        bias_min, bias_max = bias_values.min(), bias_values.max()
        margin = 0.1 * (bias_max - bias_min)
        xlim = (bias_min - margin, bias_max + margin)
        axes[idxs[0]].set_xlim(xlim)
        axes[idxs[1]].set_xlim(xlim)

    fig.supxlabel('Bias $(m^{-1})$', fontsize=12, x=0.75, y=0.060)
    fig.add_artist(plt.Line2D([0.505, 0.99], [0.105, 0.105],
                              transform=fig.transFigure,
                              color='darkgrey', linewidth=0.5))
    plt.tight_layout()
    if export:
    
    	outdir = f"../results/Comparisons/{depth}"
    	os.makedirs(outdir, exist_ok=True)
    	outfile = f"{outdir}/Comparison_{depth}_R2_MAE_Bias_sat_bgc.pdf"
    	plt.savefig(outfile)
    print(f"Figure saved to: {outfile}")
    plt.show()

def plot_sat_experiments(
    r2_depth, mae_depth, bias_depth, depth, export=True
):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import ScalarFormatter

    mld_na=-36 
    mld_stg=-50 

    # Define depth-related parameters
    if str(depth) == "50":
        depth_range = np.arange(1, 52, 2)
        yticks = (-1, -11, -21, -31, -41, -51)
    elif str(depth) == "250":
        depth_range = np.arange(1, 253, 2)
        yticks = (-1, -51, -101, -151, -201, -250)
    else:
        raise ValueError("Unsupported depth value. Use '50' or '250'.")

    # Formatter
    formatter = ScalarFormatter()
    formatter.set_scientific(True)
    formatter.set_powerlimits((-3, 2))

    fig, axes = plt.subplots(
        1, 6, figsize=(16, 5), dpi=300,
        gridspec_kw={'width_ratios': [1, 1, 0.5, 0.5, 0.5, 0.5]},
        sharey=False
    )

    colors = {'NA': 'g', 'STG': '#1f77b4'}

    # --- R² Plot ---
    ax = axes[0]
    ax.axhline(y=mld_stg, color=colors['STG'], linewidth=3, alpha=0.2, label='Mean STG MLD')
    ax.axhline(y=mld_na, color=colors['NA'], linewidth=3, alpha=0.2, label='Mean NA MLD')
    ax.plot(r2_depth["NA_S3IOPS"], -depth_range, color=colors['NA'], alpha=0.7, linestyle="--", linewidth=0.9, label="NA S3IOPS")
    ax.plot(r2_depth["NA_S3OLCI"], -depth_range, color=colors['NA'], alpha=0.7, linestyle="-", linewidth=0.9, label="NA S3OLCI")
    ax.plot(r2_depth["STG_S3IOPS"], -depth_range, color=colors['STG'], alpha=0.7, linestyle="--", linewidth=0.9, label="STG S3IOPS")
    ax.plot(r2_depth["STG_S3OLCI"], -depth_range, color=colors['STG'], alpha=0.7, linestyle="-", linewidth=0.9, label="STG S3OLCI")
    ax.set_xlabel("R\u00B2", fontsize=12)
    ax.set_ylabel("Depth (m)", fontsize=12)
    ax.set_yticks(yticks)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=8)

    # --- MAE Plot ---
    ax = axes[1]
    ax.axhline(y=mld_stg, color=colors['STG'], linewidth=3, alpha=0.2)
    ax.axhline(y=mld_na, color=colors['NA'], linewidth=3, alpha=0.2)
    ax.set_yticklabels([])
    ax.plot(mae_depth["NA_S3IOPS"], -depth_range, color=colors['NA'], alpha=0.7, linestyle="--", linewidth=0.9)
    ax.plot(mae_depth["NA_S3OLCI"], -depth_range, color=colors['NA'], alpha=0.7, linestyle="-", linewidth=0.9)
    ax.plot(mae_depth["STG_S3IOPS"], -depth_range, color=colors['STG'], alpha=0.7, linestyle="--", linewidth=0.9)
    ax.plot(mae_depth["STG_S3OLCI"], -depth_range, color=colors['STG'], alpha=0.7, linestyle="-", linewidth=0.9)
    ax.set_yticks(yticks)
    ax.set_xlabel("MAE $(m^{-1})$", fontsize=12)
    ax.tick_params(labelsize=10)
    axes[1].xaxis.set_major_formatter(formatter)

    # --- Bias bar plots ---
    for i, (region, dataset) in enumerate([
        ("NA", "S3OLCI"), ("NA", "S3IOPS"),
        ("STG", "S3OLCI"), ("STG", "S3IOPS")
    ]):
        ax = axes[i + 2]
        ax.barh(depth_range, bias_depth[f"{region}_{dataset}"], color=colors[region], alpha=0.6)
        ax.set_yticklabels([])
        ax.invert_yaxis()
        ax.axvline(x=0, color='darkgrey', linestyle='--', linewidth=1)
        ax.tick_params(labelsize=10)
        ax.set_title(dataset, fontsize=11)
        ax.xaxis.set_major_formatter(formatter)

    # Auto-scale bias plot x-limits by region
    for region, idxs in zip(["NA", "STG"], [(2, 3), (4, 5)]):
        bias_values = np.concatenate([bias_depth[f"{region}_S3OLCI"], bias_depth[f"{region}_S3IOPS"]])
        bias_min, bias_max = bias_values.min(), bias_values.max()
        margin = 0.1 * (bias_max - bias_min)
        xlim = (bias_min - margin, bias_max + margin)
        axes[idxs[0]].set_xlim(xlim)
        axes[idxs[1]].set_xlim(xlim)

    fig.supxlabel('Bias $(m^{-1})$', fontsize=12, x=0.75, y=0.060)
    fig.add_artist(plt.Line2D([0.505, 0.99], [0.105, 0.105],
                              transform=fig.transFigure,
                              color='darkgrey', linewidth=0.5))
    plt.tight_layout()
    if export:
    
    	outdir = f"../results/Comparisons/{depth}"
    	os.makedirs(outdir, exist_ok=True)
    	outfile = f"{outdir}/Comparison_{depth}_R2_MAE_Bias_sat.pdf"
    	plt.savefig(outfile)
    print(f"Figure saved to: {outfile}")
    plt.show()
