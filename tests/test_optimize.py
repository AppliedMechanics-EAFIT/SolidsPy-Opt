import unittest
import numpy as np

# Import your package and relevant functions
from solidspy_opt.optimize import ESO_stress, ESO_stiff, BESO, SIMP
from solidspy_opt.utils.structures import structure_3d, structures

class TestSolidspyOpt(unittest.TestCase):

    def test_eso_stress_2D(self):
        """
        Test ESO_stress in 2D with a small mesh.
        """
        # Create a tiny 2D structure
        L, H = 8, 8
        nx, ny = 8, 8
        dirs = np.array([[0, -1]])  # single downward load
        positions = np.array([[4, 1]])  # near top-right corner
        nodes, mats, els, loads, idx_BC = structures(L=L, H=H, nx=nx, ny=ny, dirs=dirs, positions=positions, n=1)

        # Run a short ESO_stress optimization
        ELS_opt, nodes_opt, UC, E_nodes, S_nodes = ESO_stress(
            nodes=nodes,
            els=els,
            mats=mats,
            loads=loads,
            idx_BC=idx_BC,
            niter=2,    # minimal iterations for quick test
            RR=0.1,
            ER=0.1,
            volfrac=0.5,
            plot=False, # disable plotting for tests
            dim_problem=2,
            nnodes=4
        )

        # Basic checks
        self.assertIsNotNone(ELS_opt, "Returned elements array is None.")
        self.assertGreater(ELS_opt.shape[0], 0, "No elements remain after optimization.")
        self.assertIsNotNone(nodes_opt, "Returned nodes array is None.")
        self.assertIsNotNone(UC, "No displacement array returned.")

    def test_eso_stress_3D(self):
        """
        Test ESO_stress in 3D with a small mesh.
        """
        # Create a tiny 3D structure
        L = H = W = 8
        nx = ny = nz = 8
        dirs = np.array([[0, 0, -1]])   # single downward load
        positions = np.array([[1, 1, 9]])  # near top corner
        nodes_3d, mats_3d, els_3d, loads_3d, idx_BC_3d = structure_3d(
            L, H, W, 2.0e5, 0.3, nx, ny, nz, dirs, positions
        )

        # Run a short ESO_stress optimization
        ELS_opt, nodes_opt, UC, E_nodes, S_nodes = ESO_stress(
            nodes=nodes_3d,
            els=els_3d,
            mats=mats_3d,
            loads=loads_3d,
            idx_BC=idx_BC_3d,
            niter=2,    
            RR=0.1,
            ER=0.1,
            volfrac=0.5,
            plot=False,
            dim_problem=3,
            nnodes=8
        )

        # Basic checks
        self.assertIsNotNone(ELS_opt, "Returned 3D elements array is None.")
        self.assertGreater(ELS_opt.shape[0], 0, "No 3D elements remain after optimization.")
        self.assertIsNotNone(nodes_opt, "Returned 3D nodes array is None.")
        self.assertIsNotNone(UC, "No 3D displacement array returned.")

    def test_eso_stiff_2D(self):
        """
        Test ESO_stiff in 2D with a small mesh.
        """
        # Create a tiny 2D structure
        L, H = 8, 8
        nx, ny = 8, 8
        dirs = np.array([[0, -1]])  # single downward load
        positions = np.array([[4, 1]])  # near top-right corner
        nodes, mats, els, loads, idx_BC = structures(L=L, H=H, nx=nx, ny=ny, dirs=dirs, positions=positions, n=1)

        # Run a short ESO_stiff optimization
        ELS_opt, nodes_opt, UC, E_nodes, S_nodes = ESO_stiff(
            nodes=nodes,
            els=els,
            mats=mats,
            loads=loads,
            idx_BC=idx_BC,
            niter=2,
            RR=0.1,
            ER=0.1,
            volfrac=0.5,
            plot=False,
            dim_problem=2,
            nnodes=4
        )

        # Basic checks
        self.assertIsNotNone(ELS_opt)
        self.assertGreater(ELS_opt.shape[0], 0)
        self.assertIsNotNone(nodes_opt)
        self.assertIsNotNone(UC)

    def test_eso_stiff_3D(self):
        """
        Test ESO_stiff in 3D with a small mesh.
        """
        # Create a tiny 3D structure
        L = H = W = 2
        nx = ny = nz = 2
        dirs = np.array([[0, 0, -1]])   
        positions = np.array([[1, 1, 1]])  
        nodes_3d, mats_3d, els_3d, loads_3d, idx_BC_3d = structure_3d(
            L, H, W, 2.0e5, 0.3, nx, ny, nz, dirs, positions
        )

        # Run a short ESO_stiff optimization
        ELS_opt, nodes_opt, UC, E_nodes, S_nodes = ESO_stiff(
            nodes=nodes_3d,
            els=els_3d,
            mats=mats_3d,
            loads=loads_3d,
            idx_BC=idx_BC_3d,
            niter=2,
            RR=0.1,
            ER=0.1,
            volfrac=0.5,
            plot=False,
            dim_problem=3,
            nnodes=8
        )

        self.assertIsNotNone(ELS_opt)
        self.assertGreater(ELS_opt.shape[0], 0)
        self.assertIsNotNone(nodes_opt)
        self.assertIsNotNone(UC)

    def test_beso_2D(self):
        """
        Test BESO in 2D with a small mesh.
        """
        L, H = 8, 8
        nx, ny = 8, 8
        dirs = np.array([[0, -1]])  # single downward load
        positions = np.array([[4, 1]])  # near top-right corner
        nodes, mats, els, loads, idx_BC = structures(L=L, H=H, nx=nx, ny=ny, dirs=dirs, positions=positions, n=1)

        # Run a short BESO optimization
        ELS_opt, nodes_opt, UC, E_nodes, S_nodes = BESO(
            nodes=nodes,
            els=els,
            mats=mats,
            loads=loads,
            idx_BC=idx_BC,
            niter=2,
            t=0.0001,
            ER=0.001,
            volfrac=0.5,
            plot=False,
            dim_problem=2,
            nnodes=4
        )

        self.assertIsNotNone(ELS_opt)
        self.assertGreater(ELS_opt.shape[0], 0)
        self.assertIsNotNone(nodes_opt)
        self.assertIsNotNone(UC)

    def test_beso_3D(self):
        """
        Test BESO in 3D with a small mesh.
        """
        L = H = W = 2
        nx = ny = nz = 2
        dirs = np.array([[0, 0, -1]])   
        positions = np.array([[1, 1, 1]])  
        nodes_3d, mats_3d, els_3d, loads_3d, idx_BC_3d = structure_3d(
            L, H, W, 2.0e5, 0.3, nx, ny, nz, dirs, positions
        )

        # Run a short BESO optimization
        ELS_opt, nodes_opt, UC, E_nodes, S_nodes = BESO(
            nodes=nodes_3d,
            els=els_3d,
            mats=mats_3d,
            loads=loads_3d,
            idx_BC=idx_BC_3d,
            niter=2,
            t=0.0001,
            ER=0.001,
            volfrac=0.5,
            plot=False,
            dim_problem=3,
            nnodes=8
        )

        self.assertIsNotNone(ELS_opt)
        self.assertGreater(ELS_opt.shape[0], 0)
        self.assertIsNotNone(nodes_opt)
        self.assertIsNotNone(UC)

    def test_simp_2D(self):
        """
        Test SIMP in 2D with a small mesh.
        """
        L, H = 8, 8
        nx, ny = 8, 8
        dirs = np.array([[0, -1]])  # single downward load
        positions = np.array([[4, 1]])  # near top-right corner
        nodes, mats, els, loads, idx_BC = structures(L=L, H=H, nx=nx, ny=ny, dirs=dirs, positions=positions, n=1)

        # Run SIMP for a few iterations
        rho, nodes_opt, UC, E_nodes, S_nodes = SIMP(
            nodes=nodes,
            els=els,
            mats=mats,
            loads=loads,
            idx_BC=idx_BC,
            niter=2,
            penal=3,
            volfrac=0.5,
            dimensions=[nx, ny],
            plot=False,
            dim_problem=2,
            nnodes=4
        )

        self.assertIsNotNone(rho)
        self.assertTrue(len(rho) > 0, "Density array is empty in SIMP 2D.")
        self.assertIsNotNone(nodes_opt)
        self.assertIsNotNone(UC)

    def test_simp_3D(self):
        """
        Test SIMP in 3D with a small mesh.
        """
        L = H = W = 2
        nx = ny = nz = 2
        dirs = np.array([[0, 0, -1]])   
        positions = np.array([[1, 1, 1]])  
        nodes_3d, mats_3d, els_3d, loads_3d, idx_BC_3d = structure_3d(
            L, H, W, 2.0e5, 0.3, nx, ny, nz, dirs, positions
        )

        # Run SIMP for a few iterations
        rho, nodes_opt, UC, E_nodes, S_nodes = SIMP(
            nodes=nodes_3d,
            els=els_3d,
            mats=mats_3d,
            loads=loads_3d,
            idx_BC=idx_BC_3d,
            niter=2,
            penal=3,
            volfrac=0.5,
            plot=False,
            dim_problem=3,
            nnodes=8
        )

        self.assertIsNotNone(rho)
        self.assertTrue(len(rho) > 0, "Density array is empty in SIMP 3D.")
        self.assertIsNotNone(nodes_opt)
        self.assertIsNotNone(UC)


if __name__ == "__main__":
    unittest.main()
