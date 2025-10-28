from astropy.io import fits;
import matplotlib.pyplot as plt;
import numpy as np;


with fits.open( "fits_images/dataset/10000-20000/image10824.fits" ) as hdul_model:
    data = hdul_model[ 0 ].data;

if np.any( np.isnan( data ) ):
    print( "nans!" );
if np.std( data.flatten(), ddof=1 ) <= 0:
    print( "negative std!" ) 

fig = plt.figure( figsize=(8, 8) );
gs = fig.add_gridspec(1, 1,
                    left=0.05, right=0.95, bottom=0.1, top=0.95,
                    wspace=0.5, hspace=0.5);
ax1 = fig.add_subplot( gs[ 0, 0 ] );
ax1.set_title( 'Data' );
img1 = ax1.imshow( data );
img1.set_clim( 0, 1 );
plt.show();