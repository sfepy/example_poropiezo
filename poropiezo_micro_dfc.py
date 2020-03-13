# This example implements homogenization of piezoeletric porous media.
# The mathematical model and numerical results are described in: 
#
# ROHAN E., LUKEŠ V.
# Homogenization of the fluid-saturated piezoelectric porous media.
# International Journal of Solids and Structures
# Volume 147, 15 August 2018, Pages 110-125
# https://doi.org/10.1016/j.ijsolstr.2018.05.017
#
# Run calculation of homogeized coefficients:
#
#   ./homogen.py poropiezo_example/poropiezo_micro_dfc.py
#
# The results are stored in `poropiezo_example/results` directory.
#

import sys
import numpy as nm
import os.path as osp
from sfepy.mechanics.matcoefs import stiffness_from_youngpoisson
from sfepy.homogenization.utils import coor_to_sym, define_box_regions
import sfepy.discrete.fem.periodic as per
from sfepy.discrete.fem.mesh import Mesh
import sfepy.homogenization.coefs_base as cb
from sfepy.base.base import Struct

data_dir = 'poropiezo_example'


def data_to_struct(data):
    out = {}
    for k, v in data.items():
        out[k] = Struct(name='output_data',
                        mode='cell' if v[2] == 'c' else 'vertex',
                        data=v[0],
                        var_name=v[1],
                        dofs=None)

    return out


def get_periodic_bc(var_tab, dim=3, dim_tab=None):
    if dim_tab is None:
        dim_tab = {'x': ['left', 'right'],
                   'z': ['bottom', 'top'],
                   'y': ['near', 'far']}

    periodic = {}
    epbcs = {}

    for ivar, reg in var_tab:
        periodic['per_%s' % ivar] = pers = []
        for idim in 'xyz'[0:dim]:
            key = 'per_%s_%s' % (ivar, idim)
            regs = ['%s_%s' % (reg, ii) for ii in dim_tab[idim]]
            epbcs[key] = (regs, {'%s.all' % ivar: '%s.all' % ivar},
                          'match_%s_plane' % idim)
            pers.append(key)

    return epbcs, periodic


# reconstruct displacement, and electric fields at the microscopic level,
# see Section 6.2
def recovery_micro_dfc(pb, corrs, macro):
    eps0 = macro['eps0']
    mesh = pb.domain.mesh
    regions = pb.domain.regions
    dim = mesh.dim
    Yms_map = regions['Yms'].get_entities(0)
    Ym_map = regions['Ym'].get_entities(0)

    gl = '_' + list(corrs.keys())[0].split('_')[-1]
    u1 = -corrs['corrs_p' + gl]['u'] * macro['press'][Yms_map, :]
    phi = -corrs['corrs_p' + gl]['r'] * macro['press'][Ym_map, :]

    for ii in range(2):
        u1 += corrs['corrs_k%d' % ii + gl]['u'] * macro['phi'][ii]
        phi += corrs['corrs_k%d' % ii + gl]['r'] * macro['phi'][ii]

    for ii in range(dim):
        for jj in range(dim):
            kk = coor_to_sym(ii, jj, dim)
            phi += corrs['corrs_rs' + gl]['r_%d%d' % (ii, jj)]\
                * nm.expand_dims(macro['strain'][Ym_map, kk], axis=1)
            u1 += corrs['corrs_rs' + gl]['u_%d%d' % (ii, jj)]\
                * nm.expand_dims(macro['strain'][Yms_map, kk], axis=1)

    u = macro['u'][Yms_map, :] + eps0 * u1

    mvar = pb.create_variables(['u', 'r', 'svar'])

    e_mac_Yms = [None] * macro['strain'].shape[1]

    for ii in range(dim):
        for jj in range(dim):
            kk = coor_to_sym(ii, jj, dim)
            mvar['svar'].set_data(macro['strain'][:, kk])
            mac_e_Yms = pb.evaluate('ev_volume_integrate.i2.Yms(svar)',
                                    mode='el_avg',
                                    var_dict={'svar': mvar['svar']})

            e_mac_Yms[kk] = mac_e_Yms.squeeze()

    e_mac_Yms = nm.vstack(e_mac_Yms).T[:, nm.newaxis, :, nm.newaxis]

    mvar['r'].set_data(phi)
    E_mic = pb.evaluate('ev_grad.i2.Ym(r)',
                        mode='el_avg',
                        var_dict={'r': mvar['r']}) / eps0

    mvar['u'].set_data(u1)
    e_mic = pb.evaluate('ev_cauchy_strain.i2.Yms(u)',
                        mode='el_avg',
                        var_dict={'u': mvar['u']})
    e_mic += e_mac_Yms

    out = {
        'u0': (macro['u'][Yms_map, :], 'u', 'p'),  # macro displacement
        'u1': (u1, 'u', 'p'),  # local displacement corrections, see eq. (58)
        'u': (u, 'u', 'p'),  # total displacement
        'e_mic': (e_mic, 'u', 'c'),  # micro strain field, see eq. (58)
        'phi': (phi, 'r', 'p'),  # electric potential, see eq. (57)
        'E_mic': (E_mic, 'r', 'c'),  # electric field, see eq. (58)
    }

    return data_to_struct(out)


# define homogenized coefficients and subproblems for correctors
def define(grid0=100, filename_mesh=None):
    eps0 = 0.01 / grid0

    if filename_mesh is None:
        filename_mesh = osp.join(data_dir, 'piezo_mesh_micro_dfc.vtk')

    mesh = Mesh.from_file(filename_mesh)
    n_conduct = len(nm.unique(mesh.cmesh.cell_groups)) - 2

    sym_eye = 'nm.array([1,1,0])' if mesh.dim == 2 else\
        'nm.array([1,1,1,0,0,0])'

    bbox = mesh.get_bounding_box()
    regions = define_box_regions(mesh.dim, bbox[0], bbox[1], eps=1e-3)

    regions.update({
        'Y': 'all',
        # matrix
        'Ym': 'cells of group 1',
        'Ym_left': ('r.Ym *v r.Left', 'vertex'),
        'Ym_right': ('r.Ym *v r.Right', 'vertex'),
        'Ym_bottom': ('r.Ym *v r.Bottom', 'vertex'),
        'Ym_top': ('r.Ym *v r.Top', 'vertex'),
        'Ym_far': ('r.Ym *v r.Far', 'vertex'),
        'Ym_near': ('r.Ym *v r.Near', 'vertex'),
        'Gamma_mc': ('r.Ym *v r.Yc', 'facet', 'Ym'),
        # channel / inclusion
        'Yc': 'cells of group 2',
        'Yc0': ('r.Yc -v r.Gamma_cm', 'vertex'),
        'Gamma_cm': ('r.Ym *v r.Yc', 'facet', 'Yc'),
    })

    print('number of cnonductors: %d' % n_conduct)
    regions.update({
        'Yms': ('r.Ym +v r.Ys', 'cell'),
        'Yms_left': ('r.Yms *v r.Left', 'vertex'),
        'Yms_right': ('r.Yms *v r.Right', 'vertex'),
        'Yms_bottom': ('r.Yms *v r.Bottom', 'vertex'),
        'Yms_top': ('r.Yms *v r.Top', 'vertex'),
        'Yms_far': ('r.Yms *v r.Far', 'vertex'),
        'Yms_near': ('r.Yms *v r.Near', 'vertex'),
        'Gamma_ms': ('r.Ym *v r.Ys', 'facet', 'Ym'),
        'Gamma_msc': ('r.Yms *v r.Yc', 'facet', 'Yms'),
        'Ys': (' +v '.join(['r.Ys%d' % k for k in range(n_conduct)]),
                'cell'),
    })

    options = {
        'coefs_filename': 'coefs_poropiezo_%d' % (grid0),
        'volume': {
            'variables': ['svar'],
            'expression': 'd_volume.i2.Y(svar)',
        },
        'coefs': 'coefs',
        'requirements': 'requirements',
        'output_dir': osp.join(data_dir, 'results'),
        'ls': 'ls',
        'file_per_var': True,
        'absolute_mesh_path': True,
        'multiprocessing': False,
        'recovery_hook': recovery_micro_dfc,
    }

    fields = {
        'displacement': ('real', 'vector', 'Yms', 1),
        'potential': ('real', 'scalar', 'Ym', 1),
        'sfield': ('real', 'scalar', 'Y', 1),
    }

    variables = {
        # displacement
        'u': ('unknown field', 'displacement'),
        'v': ('test field', 'displacement', 'u'),
        'Pi_u': ('parameter field', 'displacement', 'u'),
        'U1': ('parameter field', 'displacement', '(set-to-None)'),
        'U2': ('parameter field', 'displacement', '(set-to-None)'),
        # potential
        'r': ('unknown field', 'potential'),
        's': ('test field', 'potential', 'r'),
        'Pi_r': ('parameter field', 'potential', 'r'),
        'R1': ('parameter field', 'potential', '(set-to-None)'),
        'R2': ('parameter field', 'potential', '(set-to-None)'),
        # aux variable
        'svar': ('parameter field', 'sfield', '(set-to-None)'),
    }

    epbcs, periodic = get_periodic_bc([('u', 'Yms'), ('r', 'Ym')])

    mat_g_sc, mat_d_sc = eps0, eps0**2
    # BaTiO3 - Miara, Rohan, ... doi: 10.1016/j.jmps.2005.05.006
    materials = {
        'matrix': ({
            'D': {'Ym': nm.array([[1.504, 0.656, 0.659, 0, 0, 0],
                                  [0.656, 1.504, 0.659, 0, 0, 0],
                                  [0.659, 0.659, 1.455, 0, 0, 0],
                                  [0, 0, 0, 0.424, 0, 0],
                                  [0, 0, 0, 0, 0.439, 0],
                                  [0, 0, 0, 0, 0, 0.439]]) * 1e11, }
        },),
        'piezo': ({
            'g': nm.array([[0, 0, 0, 0, 11.404, 0],
                           [0, 0, 0, 0, 0, 11.404],
                           [-4.322, -4.322, 17.360, 0, 0, 0]]) / mat_g_sc,
            'd': nm.array([[1.284, 0, 0],
                           [0, 1.284, 0],
                           [0, 0, 1.505]]) * 1e-8 / mat_d_sc,
        },),
        'fluid': ({'gamma': 1.0 / 2.15e9},),
    }

    functions = {
        'match_x_plane': (per.match_x_plane,),
        'match_y_plane': (per.match_y_plane,),
        'match_z_plane': (per.match_z_plane,),
    }

    ebcs = {
        'fixed_u': ('Corners', {'u.all': 0.0}),
        'fixed_r': ('Gamma_ms', {'r.all': 0.0}),
    }

    integrals = {
        'i2': 2,
        'i5': 5,
    }

    solvers = {
        'ls': ('ls.scipy_direct', {}),
        'ns_em6': ('nls.newton', {
                   'i_max': 1,
                   'eps_a': 1e-6,
                   'eps_r': 1e-6,
                   'problem': 'nonlinear'}),
        'ns_em3': ('nls.newton', {
                   'i_max': 1,
                   'eps_a': 1e-3,
                   'eps_r': 1e-6,
                   'problem': 'nonlinear'}),
    }

    coefs = {
        # homogenized elasticity, see eq. (46)_1
        'A': {
            'requires': ['c.A1', 'c.A2'],
            'expression': 'c.A1 + c.A2',
            'class': cb.CoefEval,
        },
        'A1': {
            'status': 'auxiliary',
            'requires': ['pis_u', 'corrs_rs'],
            'expression': 'dw_lin_elastic.i2.Yms(matrix.D, U1, U2)',
            'set_variables': [('U1', ('corrs_rs', 'pis_u'), 'u'),
                              ('U2', ('corrs_rs', 'pis_u'), 'u')],
            'class': cb.CoefSymSym,
        },
        'A2': {
            'status': 'auxiliary',
            'requires': ['corrs_rs'],
            'expression': 'dw_diffusion.i2.Ym(piezo.d, R1, R2)',
            'set_variables': [('R1', 'corrs_rs', 'r'),
                              ('R2', 'corrs_rs', 'r')],
            'class': cb.CoefSymSym,
        },
        # homogenized Biot coefficient, see eq. (46)_2
        'B': {
            'requires': ['c.Phi', 'c.B1', 'c.B2'],
            'expression': 'c.B1 - c.B2 + c.Phi * %s' % sym_eye,
            'class': cb.CoefEval,
        },
        'B1': {
            'status': 'auxiliary',
            'requires': ['pis_u', 'corrs_p'],
            'expression': 'dw_lin_elastic.i2.Yms(matrix.D, U1, U2)',
            'set_variables': [('U1', 'corrs_p', 'u'),
                              ('U2', 'pis_u', 'u')],
            'class': cb.CoefSym,
        },
        'B2': {
            'status': 'auxiliary',
            'requires': ['pis_u', 'corrs_p'],
            'expression': 'dw_piezo_coupling.i2.Ym(piezo.g, U1, R1)',
            'set_variables': [('R1', 'corrs_p', 'r'),
                              ('U1', 'pis_u', 'u')],
            'class': cb.CoefSym,
        },
        # homogenized compressibility coefficient, see eq. (46)_6
        'M': {
            'requires': ['c.Phi', 'c.N'],
            'expression': 'c.N + c.Phi * %e' % materials['fluid'][0]['gamma'],
            'class': cb.CoefEval,
        },
        'N': {
            'status': 'auxiliary',
            'requires': ['corrs_p'],
            'expression': 'dw_surface_ltr.i2.Gamma_msc(U1)',
            'set_variables': [('U1', 'corrs_p', 'u')],
            'class': cb.CoefOne,
        },
        'Phi': {
            'requires': ['c.vol'],
            'expression': 'c.vol["fraction_Yc"]',
            'class': cb.CoefEval,
        },
        # volume fractions of Ym, Yc, Ys1, Ys2, ...
        'vol': {
            'regions': ['Ym', 'Yc'] + ['Ys%d' % k for k in range(n_conduct)],
            'expression': 'd_volume.i2.%s(svar)',
            'class': cb.VolumeFractions,
        },
        'eps0': {
            'requires': [],
            'expression': '%e' % eps0,
            'class': cb.CoefEval,
        },
        'filenames': {},
    }

    requirements = {
        'pis_u': {
            'variables': ['u'],
            'class': cb.ShapeDimDim,
        },
        'pis_r': {
            'variables': ['r'],
            'class': cb.ShapeDim,
        },
        # local subproblem defined by eq. (41)
        'corrs_rs': {
            'requires': ['pis_u'],
            'ebcs': ['fixed_u', 'fixed_r'],
            'epbcs': periodic['per_u'] + periodic['per_r'],
            'is_linear': True,
            'equations': {
                'eq1':
                    """dw_lin_elastic.i2.Yms(matrix.D, v, u)
                     - dw_piezo_coupling.i2.Ym(piezo.g, v, r)
                   = - dw_lin_elastic.i2.Yms(matrix.D, v, Pi_u)""",
                'eq2':
                    """
                     - dw_piezo_coupling.i2.Ym(piezo.g, u, s)
                     - dw_diffusion.i2.Ym(piezo.d, s, r)
                     = dw_piezo_coupling.i2.Ym(piezo.g, Pi_u, s)""",
            },
            'set_variables': [('Pi_u', 'pis_u', 'u')],
            'class': cb.CorrDimDim,
            'save_name': 'corrs_rs_%d' % grid0,
            'dump_variables': ['u', 'r'],
            'solvers': {'ls': 'ls', 'nls': 'ns_em3'},
        },
        # local subproblem defined by eq. (42)
        'corrs_p': {
            'requires': [],
            'ebcs': ['fixed_u', 'fixed_r'],
            'epbcs': periodic['per_u'] + periodic['per_r'],
            'is_linear': True,
            'equations': {
                'eq1':
                    """dw_lin_elastic.i2.Yms(matrix.D, v, u)
                     - dw_piezo_coupling.i2.Ym(piezo.g, v, r)
                     = dw_surface_ltr.i2.Gamma_msc(v)""",
                'eq2':
                    """
                     - dw_piezo_coupling.i2.Ym(piezo.g, u, s)
                     - dw_diffusion.i2.Ym(piezo.d, s, r)
                     = 0"""
            },
            'class': cb.CorrOne,
            'save_name': 'corrs_p_%d' % grid0,
            'dump_variables': ['u', 'r'],
            'solvers': {'ls': 'ls', 'nls': 'ns_em6'},
        },
        # local subproblem defined by eq. (43)
        'corrs_rho': {
            'requires': [],
            'ebcs': ['fixed_u', 'fixed_r'],
            'epbcs': periodic['per_u'] + periodic['per_r'],
            'is_linear': True,
            'equations': {
                'eq1':
                    """dw_lin_elastic.i2.Yms(matrix.D, v, u)
                        - dw_piezo_coupling.i2.Ym(piezo.g, v, r)
                        = 0""",
                'eq2':
                    """
                        - dw_piezo_coupling.i2.Ym(piezo.g, u, s)
                        - dw_diffusion.i2.Ym(piezo.d, s, r)
                        =
                        - dw_surface_integrate.i2.Gamma_mc(s)"""
                },
            'class': cb.CorrOne,
            'save_name': 'corrs_p_%d' % grid0,
            'dump_variables': ['u', 'r'],
            'solvers': {'ls': 'ls', 'nls': 'ns_em6'},
        },
    }

    for k in range(n_conduct):
        sk = '%d' % k
        regions.update({
            'Ys' + sk: 'cells of group %d' % (3 + k),
            'Gamma_s' + sk: ('r.Ym *v r.Ys' + sk, 'facet', 'Ym'),
        })

        materials['matrix'][0]['D'].update({
            'Ys' + sk: stiffness_from_youngpoisson(3, 200e9, 0.25),
        })

        ebcs.update({
            'fixed_r1_k_' + sk: ('Gamma_s' + sk, {'r.0': 1.0}),
            'fixed_r0_k_' + sk: ('Gamma_s' + sk, {'r.0': 0.0}),
        })

        fixed_r0_k = ['fixed_r0_k_%d' % ii for ii in range(n_conduct)
                        if not ii == k]
        # local subproblems defined for conductors, see eq. (44)
        requirements.update({
            'corrs_k' + sk: {
                'requires': ['pis_r'],
                'ebcs': ['fixed_u', 'fixed_r1_k_' + sk] + fixed_r0_k,
                'epbcs': periodic['per_u'] + periodic['per_r'],
                'is_linear': True,
                'equations': {
                    'eq1':
                        """dw_lin_elastic.i2.Yms(matrix.D, v, u)
                            - dw_piezo_coupling.i2.Ym(piezo.g, v, r)
                            = 0""",
                    'eq2':
                        """
                            - dw_piezo_coupling.i2.Ym(piezo.g, u, s)
                            - dw_diffusion.i2.Ym(piezo.d, s, r)
                            = 0"""
                    },
                'class': cb.CorrOne,
                'save_name': 'corrs_k' + sk + '_%d' % grid0,
                'dump_variables': ['u', 'r'],
                'solvers': {'ls': 'ls', 'nls': 'ns_em6'},
            },
        })

        coefs.update({
            # homogenized coefficient (46)_3
            'H' + sk: {
                'requires': ['c.H1_' + sk, 'c.H2_' + sk],
                'expression': 'c.H1_%s - c.H2_%s' % (sk, sk),
                'class': cb.CoefEval,
            },
            'H1_' + sk: {
                'status': 'auxiliary',
                'requires': ['pis_u', 'corrs_k' + sk],
                'expression': 'dw_lin_elastic.i2.Yms(matrix.D, U1, U2)',
                'set_variables': [('U1', 'corrs_k' + sk, 'u'),
                                  ('U2', 'pis_u', 'u')],
                'class': cb.CoefSym,
            },
            'H2_' + sk: {
                'status': 'auxiliary',
                'requires': ['pis_u', 'corrs_k' + sk],
                'expression': 'dw_piezo_coupling.i2.Ym(piezo.g, U1, R1)',
                'set_variables': [('R1', 'corrs_k' + sk, 'r'),
                                  ('U1', 'pis_u', 'u')],
                'class': cb.CoefSym,
            },
            # homogenized coefficient (46)_7
            'Z' + sk: {
                'requires': ['corrs_k' + sk],
                'expression': 'dw_surface_ltr.i2.Gamma_msc(U1)',
                'set_variables': [('U1', 'corrs_k' + sk, 'u')],
                'class': cb.CoefOne,
            },
        })

    return locals()
