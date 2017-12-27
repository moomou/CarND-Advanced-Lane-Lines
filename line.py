import numpy as np

# Define conversions in x and y from pixels space to meters
ym_per_pix = 30 / 720  # meters per pixel in y dimension
xm_per_pix = 3.7 / 700  # meters per pixel in x dimension


# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None

    def _fit(self, h):
        ploty = np.linspace(0, h - 1, h)

        if self.best_fit is None:
            return 0., 0., ploty

        y_eval = np.max(ploty)
        fitx = self.best_fit[0] * ploty**2 + self.best_fit[1] * ploty + self.best_fit[2]
        return fitx, y_eval, ploty

    def update(self, best_fit, allx, ally):
        if self.best_fit is None:
            prev_fit = best_fit
        else:
            prev_fit = self.best_fit
        self.best_fit = (best_fit + prev_fit) / 2

        self.allx = allx
        self.ally = ally

        self.detected = True

    def curvature(self, h):
        fitx, y_eval, ploty = self._fit(h)
        fit_cr = np.polyfit(ploty * ym_per_pix, fitx * xm_per_pix, 2)

        # Calculate the new radii of curvature
        curvature_rad = ((1 +
                          (2 * fit_cr[0] * y_eval * ym_per_pix + fit_cr[1])**2)
                         **1.5) / np.absolute(2 * fit_cr[0])

        return curvature_rad

    def base_x(self, h):
        fitx, _, _ = self._fit(h)
        return fitx[-1]
