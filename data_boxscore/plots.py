import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches

def create_spider_chart(categories, 
                        values, 
                        std_devs=None,
                        title='Spider Chart', 
                        color='blue', 
                        ax=None, 
                        min_value=None, 
                        max_value=None, 
                        graduation_levels=5, 
                        highlight_level=None, 
                        highlight_color='red',
                        highlight_linewidth=2, 
                        highlight_linestyle='--',
                        category_colors=None, 
                        display_legend:bool=False):
    """
    Creates an enhanced spider/radar chart with custom graduations, support for negative values,
    and colored category labels.
    
    Parameters:
    -----------
    categories : list
        List of category names for the chart axes
    values : list or numpy array
        Values for each category
    title : str
        Title of the chart
    color : str
        Color for the plot
    ax : matplotlib Axes, optional
        Axes to plot on. If None, a new figure is created
    min_value : float, optional
        Minimum value for the scale. If None, determined from data
    max_value : float, optional
        Maximum value for the scale. If None, determined from data
    graduation_levels : int
        Number of graduation circles/polygons to draw
    highlight_level : float, optional
        Value at which to highlight a specific graduation
    highlight_color : str
        Color for the highlighted graduation
    highlight_linewidth : float
        Line width for the highlighted graduation
    highlight_linestyle : str
        Line style for the highlighted graduation
    category_colors : list, optional
        List of colors for each category label. If None, all labels use default color.
    
    Returns:
    --------
    fig, ax : matplotlib Figure and Axes objects
    """
    # If no axes provided, create a new figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    else:
        fig = plt.gcf()
        
    # Number of variables
    N = len(categories)
    
    # Check if category_colors is provided, otherwise use default
    if category_colors is None:
        category_colors = ['black'] * N
    elif len(category_colors) < N:
        # If not enough colors provided, extend with default color
        category_colors = list(category_colors) + ['black'] * (N - len(category_colors))
    
    # Compute angle for each category
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    
    # Determine min and max values for scaling
    if min_value is None:
        min_value = np.min(values) * 1.1 if np.min(values) < 0 else 0
    
    if max_value is None:
        max_value = np.max(values) * 1.1  # Add 10% margin
        
    # Adjust values to be relative to min_value (shifting upward)
    values_transformed = np.array(values) - min_value
    range_value = max_value - min_value
    
    # Set y-limits for the transformed values
    ax.set_ylim(0, range_value)
    
    # Calculate graduation levels
    graduation_values_original = np.linspace(min_value, max_value, graduation_levels+1)
    graduation_values_transformed = graduation_values_original - min_value
    
    # Display graduation values (original values, not transformed)
    ax.set_rticks(graduation_values_transformed)
    ax.set_yticklabels([f"{v:.2f}" for v in graduation_values_original], fontsize=8)
    
    # Draw polygons for each graduation level
    for level_original, level_transformed in zip(graduation_values_original, graduation_values_transformed):
        # Check if this level should be highlighted
        if highlight_level is not None and np.isclose(level_original, highlight_level):
            ax.plot(angles, [level_transformed] * len(angles), linewidth=highlight_linewidth, 
                   linestyle=highlight_linestyle, color=highlight_color)
            # Connect points to form a polygon
            for i in range(N):
                ax.plot([angles[i], angles[(i+1) % N]], [level_transformed, level_transformed], 
                        linewidth=highlight_linewidth, linestyle=highlight_linestyle, 
                        color=highlight_color)
        else:
            # Connect points to form a polygon (not using fill to keep it transparent)
            for i in range(N):
                ax.plot([angles[i], angles[(i+1) % N]], [level_transformed, level_transformed], 
                        linewidth=0.7, linestyle='-', color='gray', alpha=0.5)
    
    # Draw axis lines with category-specific colors
    for i, angle in enumerate(angles):
        # If we have standard deviations, draw axis line as dashed
        if std_devs is not None:
            ax.plot([angle, angle], [0, range_value], linewidth=0.8, 
                   linestyle='--', color=category_colors[i], alpha=0.7)
        else:
            # Regular solid axis line when no std_devs
            ax.plot([angle, angle], [0, range_value], linewidth=1.0, 
                   color=category_colors[i], alpha=0.7)
    
    # Add standard deviation ranges if provided
    if std_devs is not None:
        for i, (angle, value, sd, cat_color) in enumerate(zip(angles, values, std_devs, category_colors)):
            # Transform to chart scale
            value_transformed = value - min_value
            
            # Calculate lower and upper bounds with SD
            lower_bound = max(0, value_transformed - sd)
            upper_bound = min(range_value, value_transformed + sd)
            
            # Draw the SD range as a thicker solid line
            ax.plot([angle, angle], [lower_bound, upper_bound], 
                   linewidth=2.5, linestyle='-', color=cat_color, alpha=0.8)
            
            # Add small horizontal lines at the ends of the SD range for better visibility
            marker_length = 0.1  # in radians
            for bound in [lower_bound, upper_bound]:
                ax.plot([angle - marker_length/2, angle + marker_length/2], [bound, bound],
                       linewidth=1.5, color=cat_color, alpha=0.8)
    
    # Make the plot circular by repeating the first value
    values_transformed_closed = np.append(values_transformed, values_transformed[0])
    angles_closed = np.append(angles, angles[0])
    
    # Plot data (using transformed values)
    ax.plot(angles_closed, values_transformed_closed, 'o-', linewidth=2, color=color, label=title)
    ax.fill(angles_closed, values_transformed_closed, alpha=0.25, color=color)
    
    # Add a special line for zero if min_value is negative
    if min_value < 0 and 0 < max_value:
        zero_level = 0 - min_value  # Transform zero to the chart's scale
        ax.plot(angles, [zero_level] * len(angles), linewidth=1.5, 
               linestyle='-', color='black', alpha=0.7)
        # Connect points to form a polygon
        for i in range(N):
            ax.plot([angles[i], angles[(i+1) % N]], [zero_level, zero_level], 
                    linewidth=1.0, linestyle='-', color='black', alpha=0.7)
    
    # Set category labels with custom colors
    ax.set_xticks(angles)
    
    # Remove default labels
    ax.set_xticklabels([])
    
    # Add colored category labels manually
    for i, (angle, category, cat_color) in enumerate(zip(angles, categories, category_colors)):
        # Calculate the position for the label (slightly outside the circle)
        label_distance = 1.02 * range_value  # 15% outside the max radius
        ha = 'center'  
        va = 'center'
        ax.text(angle, label_distance, category, color=cat_color, 
                fontsize=9, fontweight='bold', ha=ha, va=va)
    
    # Remove default grid
    ax.grid(False)
    
    # Add title and legend
    ax.set_title(title, size=15, pad=15)
    if display_legend:
        ax.legend(loc='upper right')
    
    return fig, ax