# -*- coding: utf-8 -*-
"""
Test cases for functions on ``assemutil`` module

"""
import numpy as np
import solidspy_opt
import unittest

import solidspy_opt.optimize

class TestSolidsPyOpt(unittest.TestCase):
    def setUp(self):
        # Common parameters for the tests
        self.length = 60
        self.height = 60
        self.nx = 60
        self.ny = 60
        self.dirs = np.array([[0, -1]])
        self.positions = np.array([[15, 1]])
        self.niter = 50  # Reduced for testing efficiency
        self.penal = 3
        self.volfrac = 0.5
        self.RR = 0.005
        self.ER = 0.05
        self.t = 1e-3

    def test_SIMP(self):
        """Test SIMP functionality."""
        rho = solidspy_opt.optimize.SIMP(
            self.length, 
            self.height, 
            self.nx, 
            self.ny, 
            self.dirs, 
            self.positions, 
            self.niter, 
            self.penal, 
            plot=False
        )
        self.assertIsInstance(rho, np.ndarray, "SIMP should return a numpy array.")
        self.assertEqual(rho.size, self.nx * self.ny, "Density array size should match the number of elements.")

    def test_ESO_stress(self):
        """Test ESO based on stress."""
        els, nodes = solidspy_opt.optimize.ESO_stress(
            self.length, 
            self.height, 
            self.nx, 
            self.ny, 
            self.dirs, 
            self.positions, 
            self.niter, 
            self.RR, 
            self.ER, 
            self.volfrac, 
            plot=False
        )
        self.assertIsInstance(els, np.ndarray, "ESO_stress should return an array of elements.")
        self.assertIsInstance(nodes, np.ndarray, "ESO_stress should return an array of nodes.")
        self.assertTrue(els.size > 0, "The element array should not be empty.")

    def test_ESO_stiff(self):
        """Test ESO based on stiffness."""
        els, nodes = solidspy_opt.optimize.ESO_stiff(
            self.length, 
            self.height, 
            self.nx, 
            self.ny, 
            self.dirs, 
            self.positions, 
            self.niter, 
            self.RR, 
            self.ER, 
            self.volfrac, 
            plot=False
        )
        self.assertIsInstance(els, np.ndarray, "ESO_stiff should return an array of elements.")
        self.assertIsInstance(nodes, np.ndarray, "ESO_stiff should return an array of nodes.")
        self.assertTrue(els.size > 0, "The element array should not be empty.")

    def test_BESO(self):
        """Test Bi-directional ESO."""
        els, nodes = solidspy_opt.optimize.BESO(
            self.length, 
            self.height, 
            self.nx, 
            self.ny, 
            self.dirs, 
            self.positions, 
            self.niter, 
            self.t, 
            self.ER, 
            self.volfrac, 
            plot=False
        )
        self.assertIsInstance(els, np.ndarray, "BESO should return an array of elements.")
        self.assertIsInstance(nodes, np.ndarray, "BESO should return an array of nodes.")
        self.assertTrue(els.size > 0, "The element array should not be empty.")

if __name__ == "__main__":
    unittest.main()
