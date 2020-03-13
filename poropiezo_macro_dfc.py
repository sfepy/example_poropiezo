# This example implements homogenization of piezoeletric porous media.
# The mathematical model and numerical results are described in: 
#
# ROHAN E., LUKEŠ V.
# Homogenization of the fluid-saturated piezoelectric porous media.
# International Journal of Solids and Structures
# Volume 147, 15 August 2018, Pages 110-125
# https://doi.org/10.1016/j.ijsolstr.2018.05.017
#
# Run simulation:
#
#   ./simple.py poropiezo/poropiezo_macro_dfc.py
#
# The results are stored in `poropiezo/results` directory.
#

import sys
import numpy as nm
import os.path as osp
from sfepy import data_dir
from sfepy.base.base import Struct
from sfepy.homogenization.micmac import get_homog_coefs_linear
from sfepy.homogenization.recovery import recover_micro_hook_eps

data_dir = 'poropiezo'


def set_grad(ts, coors, mode=None, problem=None, **kwargs):
    if mode == 'qp':
        out = problem.data.reshape((coors.shape[0], 1, 1))
        return {'cs': out}


# projection of values from integration points into mesh vertices
def linear_projection(pb, cval):
    from sfepy.discrete import (FieldVariable, Material, Integral,
                                Equation, Equations, Problem)
    from sfepy.discrete.fem import Mesh, FEDomain, Field
    from sfepy.terms import Term
    from sfepy.solvers.ls import ScipyDirect
    from sfepy.solvers.nls import Newton
    from sfepy.base.base import IndexedStruct

    mesh = Mesh.from_file(pb.conf.filename_mesh)
    domain = FEDomain('domain', mesh)
    omega = domain.create_region('Omega', 'all')
    field = Field.from_args('scf', nm.float64, 'scalar', omega,
                            approx_order=1)

    g = FieldVariable('g', 'unknown', field)
    f = FieldVariable('f', 'test', field, primary_var_name='g')

    integral = Integral('i', order=2)
    m = Material('m', function=set_grad)

    t1 = Term.new('dw_volume_dot(f, g)', integral, omega, f=f, g=g)
    t2 = Term.new('dw_volume_lvf(m.cs, f)',
                  integral, omega, m=m, f=f)
    eq = Equation('balance', t1 - t2)
    eqs = Equations([eq])
    ls = ScipyDirect({})

    nls_status = IndexedStruct()
    nls = Newton({'eps_a': 1e-15}, lin_solver=ls, status=nls_status)
    pb = Problem('elasticity', equations=eqs)
    pb.set_solver(nls)

    out = nm.empty((g.n_dof, cval.shape[2]), dtype=nm.float64)
    for ii in range(cval.shape[2]):
        pb.data = nm.ascontiguousarray(cval[:, :, ii, :])
        pb.time_update()
        state = pb.solve()
        out[:, ii] = state.get_parts()['g']

    return out


# recover microsccopic fields in a given region, see Section 6.2
def recover_micro(pb, state):
    rreg = pb.domain.regions['Recovery']

    state_dict = state.get_parts()
    strain_qp = pb.evaluate('ev_cauchy_strain.i2.Omega(u)', mode='qp')
    press_qp = pb.evaluate('ev_volume_integrate.i2.Omega(p)', mode='qp')
    pressg_qp = pb.evaluate('ev_grad.i2.Omega(p)', mode='qp')

    dim = rreg.dim
    displ = state_dict['u']

    nodal_data = {
        'u': displ.reshape((displ.shape[0] // dim, dim)),
        'press': linear_projection(pb, press_qp),
        'strain': linear_projection(pb, strain_qp),
        'dp': linear_projection(pb, pressg_qp),
    }
    const_data = {
        'phi': pb.conf.phi,
    }

    pvar = pb.create_variables(['svar'])
    def_args = {
        'grid0': pb.conf.grid0,
        'filename_mesh': pb.conf.filename_mesh_micro,
    }
    recover_micro_hook_eps(pb.conf.poroela_micro_file, rreg,
                           pvar['svar'], nodal_data, const_data, pb.conf.eps0,
                           recovery_file_tag='_%d' % pb.conf.grid0,
                           define_args=def_args)


# evaluate macroscopic strain and export to output
def post_process(out, pb, state, extend=False):
    strain = pb.evaluate('ev_cauchy_strain.i2.Omega(u)', mode='el_avg')

    out['e'] = Struct(name='output_data',
                      mode='cell',
                      dofs=None,
                      var_name='u',
                      data=strain)

    if pb.conf.options.get('recover_micro', False):
        recover_micro(pb, state)

    return out


def coefs2qp(coefs, nqp):
    out = {}
    for k, v in coefs.items():
        if type(v) not in [nm.ndarray, float]:
            continue

        if type(v) is nm.ndarray:
            if len(v.shape) >= 3:
                out[k] = v

        out[k] = nm.tile(v, (nqp, 1, 1))

    return out


# get homogenized coefficients, recalculate them if necessary
def get_homog(coors, mode, pb, micro_filename, **kwargs):
    if not (mode == 'qp'):
        return

    nqp = coors.shape[0]
    coefs_filename = 'coefs_poropiezo_%d' % pb.conf.grid0
    coefs_filename = osp.join(pb.conf.options.get('output_dir', '.'),
                              coefs_filename) + '.h5'

    def_args = {
        'grid0': pb.conf.grid0,
        'filename_mesh': pb.conf.filename_mesh_micro,
    }
    coefs = get_homog_coefs_linear(0, 0, None,
                                   micro_filename=micro_filename,
                                   coefs_filename=coefs_filename,
                                   define_args=def_args)

    for k in coefs.keys():
        v = coefs[k]
        if type(v) is nm.ndarray:
            if len(v.shape) == 0:
                coefs[k] = v.reshape((1, 1))
            elif len(v.shape) == 1:
                coefs[k] = v[:, nm.newaxis]
        elif isinstance(v, float):
            coefs[k] = nm.array([[v]])

    out = coefs2qp(coefs, nqp)

    phi = pb.conf.phi
    Hf, Zf = 0, 0
    for ii in range(2):
        Hf += out['H%d' % ii] * phi[ii]
        Zf += out['Z%d' % ii] * phi[ii]

    out['Hf'] = Hf
    out['Zf'] = Zf

    return out


def define(grid0=16, bcmode='example'):
    eps0 = 0.01 / grid0

    phi = nm.array([-1, 1]) * 1e3

    poroela_micro_file = osp.join(data_dir, 'poropiezo_micro_dfc.py')

    fields = {
        'displacement': ('real', 'vector', 'Omega', 1),
        'pressure': ('real', 'scalar', 'Omega', 0),
        'sfield': ('real', 'scalar', 'Omega', 1),
    }

    variables = {
        'u': ('unknown field', 'displacement'),
        'v': ('test field', 'displacement', 'u'),
        'p': ('unknown field', 'pressure'),
        'q': ('test field', 'pressure', 'p'),
        'U': ('parameter field', 'displacement', 'u'),
        'P': ('parameter field', 'pressure', 'p'),
        'svar': ('parameter field', 'sfield', 'p'),
    }

    functions = {
        'get_homog': (lambda ts, coors, mode=None, problem=None, **kwargs:\
            get_homog(coors, mode, problem, poroela_micro_file, **kwargs),),
    }

    materials = {
        'hom': 'get_homog',
    }

    integrals = {
        'i2': 2,
    }

    solvers = {
        'ls': ('ls.scipy_direct', {}),
        'newton': ('nls.newton',
                   {'i_max': 10,
                    'eps_a': 1e-1,
                    'eps_r': 1e-3,
                    'problem': 'nonlinear',
                    })
    }

    options = {
        'output_dir': 'output',
        'nls': 'newton',
        'absolute_mesh_path': True,
        'recover_micro': True,
        'post_process_hook': 'post_process',
    }

    regions = {
        'Omega': 'all',
        'Left': ('vertices in (x < 0.00001)', 'facet'),
        'Right': ('vertices in (x > 0.00999)', 'facet'),
        'Top': ('vertices in (z > 0.00999)', 'facet'),
        'Bottom': ('vertices in (z < 0.00001)', 'facet'),
        'Far': ('vertices in (y > 0.00249)', 'facet'),
        'Near': ('vertices in (y < 0.00001)', 'facet'),
        'Recovery': ('vertices in (x > 0.008) & (z > 0.008) & (y < 0.0013)', 'cell'),
    }

    filename_mesh = osp.join(data_dir, 'piezo_mesh_macro.vtk')
    filename_mesh_micro = osp.join(data_dir, 'piezo_mesh_micro_dfc.vtk')

    ebcs = {
        'fixed_u_left': ('Left', {'u.all': 0.0}),
    }
    epbcs = {}

    # equations (47)
    equations = {
        'balance_of_forces': """
            dw_lin_elastic.i2.Omega(hom.A, v, u)
          - dw_biot.i2.Omega(hom.B, v, p)
          =
          - dw_lin_prestress.i2.Omega(hom.Hf, v)""",
        'mass_conservation': """
           - dw_biot.i2.Omega(hom.B, u, q)
           - dw_volume_dot.i2.Omega(hom.M, q, p)
           =
           - dw_volume_integrate.i2.Omega(hom.Zf, q)"""
    }

    return locals()
