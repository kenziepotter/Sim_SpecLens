#!/usr/bin/env python

import os, sys, time, pdb
import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table
import fitsio

def maggen(mag, fratio=1):
    newmag = -2.5*np.log10(fratio)+mag
    return newmag

def wrap_redrock(info, mp=1, prefix='lenssource'):
    """Simple wrapper on redrock given a simulation table."""

    from redrock.external.desi import rrdesi

    os.environ['OMP_NUM_THREADS'] = '1'

    nchunk = len(set(info['CHUNK']))
    zbestfiles = []

    for ichunk in set(info['CHUNK']):
        specfile = '{}-spectra-chunk{:03d}.fits'.format(prefix, ichunk)
        zbestfile = specfile.replace('-spectra-', '-zbest-')
        rrfile = zbestfile.replace('.fits', '.h5')
        if os.path.isfile(rrfile):
            os.remove(rrfile)
        zbestfiles.append(zbestfile)
        print('Writing redshifts for chunk {}/{} to {}'.format(ichunk, nchunk-1, zbestfile))

        rrdesi(options=['--zbest', zbestfile, '--mp', str(mp), '--output', rrfile, specfile])

    # Gather up all the results and write out a summary table.    
    zbest = Table(np.hstack([fitsio.read(zbestfile) for zbestfile in zbestfiles]))
    zbest = zbest[np.argsort(zbest['TARGETID'])]

    infofile = '{}-zbest.fits'.format(prefix)
    print('Writing {} redshifts to {}'.format(len(zbest), infofile))
    zbest.write(infofile, overwrite=True)

    return zbest

def sim_source_spectra(allinfo, allzbest, infofile='source-truth.fits', debug=False):
    """Build the residual (source) spectra. No redshift-fitting.

    """
    from desispec.io import read_spectra, write_spectra
    from desispec.spectra import Spectra
    from desispec.interpolation import resample_flux
    from desispec.resolution import Resolution

    from redrock.external.desi import DistTargetsDESI
    from redrock.templates import find_templates, Template

    assert(np.all(allinfo['TARGETID'] == allzbest['TARGETID']))

    nsim = len(allinfo)

    # Select the subset of objects for which we got the correct lens (BGS)
    # redshift.
    these = np.where((allzbest['SPECTYPE'] == 'GALAXY') *
                     (np.abs(allzbest['Z'] - allinfo['LENS_Z']) < 0.003))[0]
    print('Selecting {}/{} lenses with the correct redshift'.format(len(these), nsim))
    if len(these) == 0:
        raise ValueError('No spectra passed the cuts!')

    allinfo = allinfo[these]
    allzbest = allzbest[these]

    print('Writing {}'.format(infofile))
    allinfo.write(infofile, overwrite=True)

    tempfile = find_templates()[0]
    rrtemp = Template(tempfile)
    
    # loop on each chunk of lens+source spectra
    nchunk = len(set(allinfo['CHUNK']))
    for ichunk in set(allinfo['CHUNK']):

        I = np.where(allinfo['CHUNK'] == ichunk)[0]
        info = allinfo[I]
        zbest = allzbest[I]
        
        specfile = 'lenssource-spectra-chunk{:03d}.fits'.format(ichunk)
        sourcefile = 'source-spectra-chunk{:03d}.fits'.format(ichunk)
        spectra = read_spectra(specfile).select(targets=info['TARGETID'])
        for igal, zz in enumerate(zbest):
            zwave = rrtemp.wave * (1 + zz['Z'])
            zflux = rrtemp.flux.T.dot(zz['COEFF']).T #/ (1 + zz['Z'])
            if debug:
                fig, ax = plt.subplots()
            for band in spectra.bands:
                R = Resolution(spectra.resolution_data[band][igal])
                # use fastspecfit here
                modelflux = R.dot(resample_flux(spectra.wave[band], zwave, zflux))
                if debug:
                    ax.plot(spectra.wave[band], spectra.flux[band][igal, :])
                    ax.plot(spectra.wave[band], modelflux)
                    ax.set_ylim(np.median(spectra.flux['r'][igal, :]) + np.std(spectra.flux['r'][igal, :]) * np.array([-1.5, 3]))
                    #ax.set_xlim(4500, 5500)
                spectra.flux[band][igal, :] -= modelflux # subtract
            if debug:
                qafile = 'source-spectra-chunk{:03d}-{}.png'.format(ichunk, igal)
                fig.savefig(qafile)
                plt.close()

        print('Writing {} spectra to {}'.format(len(zbest), sourcefile))
        write_spectra(outfile=sourcefile, spec=spectra)

    return allinfo    

def sim_lenssource_spectra(BGSmags, fratios, seed=None, exptime=1000., nperchunk=500,
                           infofile='lenssource-truth.fits', debug=False):
    """Build the (noisy) lens+source spectra. No redshift-fitting.

    """
    from astropy.io import fits
    from desisim.templates import BGS, ELG
    from desisim.scripts.quickspectra import sim_spectra
    from desisim.io import read_basis_templates
    from desispec.io import read_spectra
    
    rand = np.random.RandomState(seed)
    nsim = len(BGSmags)
    assert(nsim == len(fratios))

    if nperchunk > 500:
        raise ValueError('nperchunk={} exceeds the maximum number allowed by redrock'.format(nperchunk))

    nchunk = np.ceil(nsim / nperchunk).astype(int)

    # [1] Build the noise-less lens (BGS) spectra.

    # Read one healpix of the Buzzard mocks for redshift distribution.
    mockfile = os.path.join(os.getenv('DESI_ROOT'), 'mocks', 'buzzard', 'buzzard_v1.6_desicut',
                            '8', '0', '0', 'Buzzard_v1.6_lensed-8-0.fits')
    print('Reading {}'.format(mockfile))
    mock_BGS = Table(fitsio.read(mockfile)) #columns='lmag z'.split()
    
    # From the BGS template library, select a reddish galaxy
    tflux, twave, tmeta_BGS = read_basis_templates('BGS')
    i = np.argmin(np.abs(2.0 - tmeta_BGS['D4000']))
    iredBGS = i
    redspecBGS = tflux[i, :]
    Itempl_BGS = np.array([iredBGS])

    # LMAG: observed mag, DECam grizY
    mock_mag_r = mock_BGS['LMAG'][:, 1] # r-band
    dm = 0.01
    zz_BGS = np.zeros_like(BGSmags)
    for ii, mag in enumerate(BGSmags):
        I = np.flatnonzero(np.abs(mock_mag_r - mag) <= dm)
        zz_BGS[ii] = mock_BGS['Z'][rand.choice(I, size=1, replace=False)]

    input_meta_BGS = Table()
    input_meta_BGS['TEMPLATEID'] = [Itempl_BGS]*nsim
    input_meta_BGS['SEED'] = np.arange(nsim) # [seed]*nsim # 
    input_meta_BGS['REDSHIFT'] = zz_BGS
    input_meta_BGS['MAG'] = BGSmags
    input_meta_BGS['MAGFILTER'] = ['decam2014-r']*nsim

    BGSflux, BGSwave, BGSmeta, BGSobjmeta = BGS().make_templates(
        input_meta=input_meta_BGS, nocolorcuts=True, seed=seed)

    # [2] Build the noise-less source (ELG) spectra.
    #ELGmags = maggen(BGSmags[:, np.newaxis], fratios[np.newaxis, :])
    ELGmags = maggen(BGSmags, fratios)

    # Select a single ELG template.
    tflux, twave, tmeta_ELG = read_basis_templates('ELG')
    i = np.argmin(np.abs(1.0 - tmeta_ELG['D4000'])) # MIGHT NEED TO ADJUST THIS LINE 
    iblueELG = i
    bluespecELG = tflux[i, :]
    Itempl_ELG = np.array([iblueELG])

    # uncorrelated redshifts
    zmin_ELG, zmax_ELG = 0.8, 1.4
    zz_ELG = rand.uniform(zmin_ELG, zmax_ELG, nsim)
    
    input_meta_ELG = Table()
    input_meta_ELG['TEMPLATEID'] = [Itempl_ELG]*nsim
    input_meta_ELG['SEED'] = [3]*nsim # [seed]*nsim # np.arange(nsim) hack!
    input_meta_ELG['REDSHIFT'] = zz_ELG
    input_meta_ELG['MAG'] = ELGmags
    input_meta_ELG['MAGFILTER'] = ['decam2014-r']*nsim

    ELGflux, ELGwave, ELGmeta, ELGobjmeta = ELG().make_templates(
        input_meta=input_meta_ELG, nocolorcuts=True, seed=seed)
    assert(np.all(BGSwave == ELGwave))

    # Pack the simulation info into a table, for convenience.
    siminfo = Table()
    siminfo['TARGETID'] = np.arange(nsim, dtype=np.int64)
    siminfo['LENS_Z'] = input_meta_BGS['REDSHIFT'].astype('f4')
    siminfo['LENS_MAG'] = input_meta_BGS['MAG'].astype('f4')
    siminfo['SOURCE_Z'] = input_meta_ELG['REDSHIFT'].astype('f4')
    siminfo['SOURCE_MAG'] = input_meta_ELG['MAG'].astype('f4')
    siminfo['FRATIO'] = fratios.astype('f4')
    siminfo['CHUNK'] = np.zeros(nsim, dtype=np.int32)

    # Generate simulated DESI spectra given real spectra and observing
    # conditions. Divide the sample into chunks with a fixed number of
    # spectra per chunk (but no more than 500).
    obscond = {'AIRMASS': 1.3, 'EXPTIME': exptime, 'SEEING': 1.1,
               'MOONALT': -60, 'MOONFRAC': 0.0, 'MOONSEP': 180}

    simflux = BGSflux + ELGflux
    simwave = BGSwave

    for ichunk in np.arange(nchunk):
        specfile = 'lenssource-spectra-chunk{:03d}.fits'.format(ichunk)
        print('Writing chunk {}/{} to {}'.format(ichunk, nchunk-1, specfile))
        i1 = ichunk * nperchunk
        i2 = (ichunk+1) * nperchunk
        siminfo['CHUNK'][i1:i2] = ichunk
        sim_spectra(simwave, simflux[i1:i2, :], 'dark', specfile, obsconditions=obscond,
                    sourcetype='bgs', seed=seed, targetid=siminfo['TARGETID'][i1:i2],
                    redshift=siminfo['LENS_Z'][i1:i2])
        if debug:
            spectra = read_spectra(specfile)
            for igal in np.arange(spectra.num_targets()):
                qafile = 'lenssource-spectra-chunk{:03d}-{}.png'.format(ichunk, igal)
                fig, ax = plt.subplots()
                for band in spectra.bands:
                    ax.plot(spectra.wave[band], spectra.flux[band][igal, :])
                ax.plot(simwave, simflux[i1:i2, :][igal, :], color='k', lw=2)
                ax.set_ylim(np.median(simflux[i1:i2, :][igal, :]) + np.std(spectra.flux['r'][igal, :]) * np.array([-1.5, 3]))
                fig.savefig(qafile)
                plt.close()
                
    # write out and return
    hduflux = fits.PrimaryHDU(simflux)
    hduflux.header['EXTNAME'] = 'FLUX'
    hduflux.header['BUNIT'] = '10^(-17) erg/(s cm2 Angstrom)'

    hdubgs = fits.ImageHDU(BGSflux)
    hdubgs.header['EXTNAME'] = 'BGSFLUX'

    hduelg = fits.ImageHDU(ELGflux)
    hduelg.header['EXTNAME'] = 'ELGFLUX'

    hduwave = fits.ImageHDU(simwave)
    hduwave.header['EXTNAME'] = 'WAVE'
    hduwave.header['BUNIT'] = 'Angstrom'
    hduwave.header['AIRORVAC'] = ('vac', 'vacuum wavelengths')

    hdutable = fits.convenience.table_to_hdu(siminfo)
    hdutable.header['EXTNAME'] = 'METADATA'

    hx = fits.HDUList([hduflux, hdubgs, hduelg, hduwave, hdutable])

    print('Writing {}'.format(infofile))
    hx.writeto(infofile, overwrite=True)

    return siminfo

def main():
    """Simulate and Combine spectra.  We take n BGS spectra and n flux ratios and
    combine them to find n^2 ELG background spectra. Then we find the redshift
    and templates of the (hopefully) front galaxy.

    """
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--nsim', help='Number of spectra to simulate.', type=int, default=10)
    parser.add_argument('--nperchunk', help='Number of spectra per chunk.', type=int, default=500)
    parser.add_argument('--seed', help='Seed', type=int, default=1)
    parser.add_argument('--mp', help='Number of multiprocessing cores', type=int, default=32)
    parser.add_argument('--exptime', help='Exposure time', type=float, default=1000.)
    parser.add_argument('--debug', help='Generate debugging QA.', default=False, action='store_true')
    parser.add_argument('--overwrite', help='Overwrite existing files.', default=False, action='store_true')
    args = parser.parse_args()

    start = time.time()

    rand = np.random.RandomState(args.seed)

    # BGS magnitude prior distribution
    minMag = 18.0
    maxMag = 20.0
    BGSmags =  rand.uniform(maxMag, minMag, args.nsim)

    # flux ratio prior distribution
    minF = 0.001
    maxF = 0.3
    #fratios = np.logspace(np.log10(minF), np.log10(maxF), num)
    fratios = rand.uniform(np.sqrt(minF), np.sqrt(maxF), args.nsim)**2

    # Build the lens and source spectra if they don't already exist.
    infofile = 'lenssource-truth.fits'
    if args.overwrite or not os.path.isfile(infofile):
        lenssource_info = sim_lenssource_spectra(BGSmags, fratios, seed=args.seed,
                                                 exptime=args.exptime,
                                                 nperchunk=args.nperchunk,
                                                 infofile=infofile,
                                                 debug=args.debug)
    else:
        lenssource_info = Table(fitsio.read(infofile, ext='METADATA'))
        print('Read properties for {} simulated lens+source spectra from {}'.format(
            len(lenssource_info), infofile))

    # Derive lens+source redshifts if they don't already exist.
    infofile = 'lenssource-zbest.fits'
    if args.overwrite or not os.path.isfile(infofile):
        lenssource_zbest = wrap_redrock(lenssource_info, mp=args.mp, prefix='lenssource')
    else:
        lenssource_zbest = Table(fitsio.read(infofile))
        print('Read redshifts for {} simulated spectra from {}'.format(
            len(lenssource_zbest), infofile))

    # Build the residual (source-only) spectra and re-run redrock, but only on
    # the subset of objects for which we got the correct lens (BGS) redshift.
    infofile = 'source-truth.fits'
    if args.overwrite or not os.path.isfile(infofile):
        source_info = sim_source_spectra(lenssource_info, lenssource_zbest,
                                         infofile=infofile, debug=args.debug)
    else:
        source_info = Table(fitsio.read(infofile))
        print('Read properties for {} source spectra from {}'.format(
            len(source_info), infofile))

    # Derive source redshifts if they don't already exist.
    infofile = 'source-zbest.fits'
    if args.overwrite or not os.path.isfile(infofile):
        source_zbest = wrap_redrock(source_info, mp=args.mp, prefix='source')

    print('All done in {:.3f} min'.format((time.time() - start) / 60))
    
    #pdb.set_trace()
    
if __name__ == '__main__':
    sys.exit(main())
