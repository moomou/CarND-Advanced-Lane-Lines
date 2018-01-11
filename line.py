import numpy as np

import util

# Define conversions in x and y from pixels space to meters
ym_per_pix = 15 / 720  # meters per pixel in y dimension
xm_per_pix = 3.7 / 950  # meters per pixel in x dimension


# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # past best_fits
        self.past_fits = []
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # radius of curvature of the line in some units
        self.radius_of_curvature = []
        # distance in meters of vehicle center from the line
        self.line_base_pos = []
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
            diff = None
        else:
            diff = best_fit - self.best_fit

        if diff is not None and np.sum(np.square(diff)) > 20e3:
            print('Diff::', np.sum(np.square(diff)))
            self.detected = False
            return
        else:
            self.past_fits.append(best_fit)
            self.past_fits = self.past_fits[-8:]

        self.detected = True
        if len(self.past_fits) > 1:
            self.best_fit = np.mean(np.array(self.past_fits), axis=0)
        else:
            self.best_fit = self.past_fits[0]

        self.allx = allx
        self.ally = ally

    def curvature(self, h):
        fitx, y_eval, ploty = self._fit(h)
        fit_cr = np.polyfit(ploty * ym_per_pix, fitx * xm_per_pix, 2)

        # Calculate the new radii of curvature
        curvature_rad = ((1 +
                          (2 * fit_cr[0] * y_eval * ym_per_pix + fit_cr[1])**2)
                         **1.5) / np.absolute(2 * fit_cr[0])

        curvatures = self.radius_of_curvature
        curvatures.append(curvature_rad)
        selected = util.reject_outliers(np.array(curvatures))

        self.curvatures = np.array(curvatures)[selected].tolist()
        self.curvatures = self.curvatures[-5:]

        return np.mean(self.radius_of_curvature)

    def base_x(self, h):
        fitx, _, _ = self._fit(h)

        line_base_pos = self.line_base_pos
        line_base_pos.append(fitx[-1])

        self.line_base_pos = np.array(line_base_pos).tolist()
        self.line_base_pos = self.line_base_pos[-5:]

        return np.mean(self.line_base_pos)
