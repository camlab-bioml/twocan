from typing import Tuple, List, Optional
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.axes import Axes
from skimage import exposure
import numpy as np


def get_rectangle_area(w1: float, h1: float, M: np.ndarray) -> Tuple[float, float, float]:
    """Calculate the area and dimensions of a transformed rectangle.
    
    Parameters
    ----------
    w1 : float
        Width of original rectangle.
    h1 : float
        Height of original rectangle.
    M : np.ndarray
        2x3 affine transformation matrix.
        
    Returns
    -------
    Tuple[float, float, float]
        Area, x-length, and y-length of transformed rectangle.
    """
    original_rectangle = np.array([[0, 0],[w1, 0],[w1, h1], [0, h1], [0, 0]])   
    transformed_rectangle = np.dot(original_rectangle, M[:2, :2].T) + M[:2, 2]
    # Calculate area using Shoelace formula
    x = transformed_rectangle[:, 0]
    y = transformed_rectangle[:, 1]
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    x_length = np.max(x) - np.min(x)
    y_length = np.max(y) - np.min(y)
    return area, x_length, y_length


def plot_cartoon_affine(w1: float, h1: float, M: np.ndarray, w2: float, h2: float, 
                       ax: Optional[Axes] = None, show_source: bool = False, 
                       source_color: str = 'green', target_color: str = 'purple') -> Tuple[Axes, List[Line2D]]:
    """Plot a cartoon representation of an affine transformation.
    
    Visualizes how a rectangle is transformed by an affine matrix, useful for
    understanding registration transformations.
    
    Parameters
    ----------
    w1, h1 : float
        Width and height of source rectangle.
    M : np.ndarray
        2x3 affine transformation matrix.
    w2, h2 : float
        Width and height of target rectangle.
    ax : Optional[Axes], default=None
        Matplotlib axes for plotting. If None, current axes will be used.
    show_source : bool, default=False
        Whether to show the original source rectangle.
    source_color : str, default='green'
        Color for source rectangle and its transformation.
    target_color : str, default='purple'
        Color for target rectangle.
        
    Returns
    -------
    Tuple[Axes, List[Line2D]]
        The matplotlib axes object and list of plotted lines.
        
    Examples
    --------
    >>> plot_cartoon_affine(*images['IMC'].shape[-2:], reg.M_, *images['IF'].shape[-2:])
    """
    if ax is None:
        ax = plt.gca()
    
    # Define the vertices of the original rectangle
    original_rectangle = np.array([[0, 0],[w1, 0],[w1, h1], [0, h1], [0, 0]])
    target = np.array([[0, 0],[w2, 0],[w2, h2], [0, h2], [0, 0]])
    # Apply the transformation to the rectangle
    transformed_rectangle = np.dot(original_rectangle, M[:2, :2].T) + M[:2, 2]
    area = get_rectangle_area(w1, h1, M)[0]
    
    # Plot the rectangles
    lines = []
    if show_source:
        lines.append(ax.plot(original_rectangle[:, 0], original_rectangle[:, 1], 
                            color=source_color, linestyle='--', label='Source')[0])
    lines.append(ax.plot(transformed_rectangle[:, 0], transformed_rectangle[:, 1], 
                        color=source_color, label='Source transformed')[0])
    lines.append(ax.plot(target[:, 0], target[:, 1], 
                        color=target_color, label='Target')[0])
    
    ax.set_aspect('equal')
    ax.set_title(f'Transformed area: {area:.2f}')
    
    return ax, lines


def get_merge(source: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge two images into a color-coded overlay.
    
    Creates a visualization where the source image is shown in green and the
    target image in magenta, with overlapping regions appearing white.
    
    Parameters
    ----------
    source : np.ndarray
        Source image array.
    target : np.ndarray
        Target image array.
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Three RGBA arrays: green channel (source), magenta channel (target),
        and their additive combination.
    """
    # Stretch the intensity range of both images
    source_stretched = exposure.rescale_intensity(source, out_range=(0, 1))
    target_stretched = exposure.rescale_intensity(target, out_range=(0, 1))
    green = np.zeros((*source_stretched.shape, 4))
    green[..., 0] = 0  # R
    green[..., 1] = source_stretched  # G
    green[..., 2] = 0  # B
    green[..., 3] = 1  # Alpha
    magenta = np.zeros((*target_stretched.shape, 4))
    magenta[..., 0] = target_stretched  # R
    magenta[..., 1] = 0  # G
    magenta[..., 2] = target_stretched  # B
    magenta[..., 3] = 1  # Alpha
    # Combine images additively
    comb = np.clip(green + magenta, 0, 1)
    return (green, magenta, comb)
