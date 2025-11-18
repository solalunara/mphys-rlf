from pathlib import Path, PurePath;
import re;
from pybdsf_analysis.recursive_file_analyzer import RecursiveFileAnalyzer;
import logging;
import utils.paths;
import logging;

class LogAnalyzer( RecursiveFileAnalyzer ):
    """
    A class to analyze PyBDSF log files

    Parameters
    ----------
    subdir : PurePath | str
        The subdirectory of this object, e.g. 'generated' or 'dataset'
    log_file_dir : Path = utils.paths.PYBDSF_LOG_PARENT
        The directory of the log files, without the subdirectory
    log_level : int = logging.INFO
    """
    def __init__( self, subdir: PurePath | str, log_file_dir: Path = utils.paths.PYBDSF_LOG_PARENT, log_level: int = logging.INFO ):
        super().__init__( log_file_dir, log_level );
        self.subdir = subdir if isinstance( subdir, PurePath ) else PurePath( subdir );

    def get_flux( self, path: Path ):
        """
        A function to get the flux of a log file at path.

        Parameters
        ----------
        path: Path
            The path to the pybdsf log file

        Returns
        -------
        flux: float
            The flux of the image in Jy or arbitrary units (because of 0-1 normalizaiton)
        """
        with open( str( path ) ) as file:
            filedata = file.read();
        exp = re.compile( r"Flux from sum of \(non-blank\) pixels ..... : (\d+\.\d+) Jy" );
        match = exp.search( filedata );
        flux = float( match.group( 1 ) );
        return flux;

    def get_mean( self, path: Path ):
        """
        A function to get the mean of a log file at path.

        Parameters
        ----------
        path: Path
            The path to the pybdsf log file

        Returns
        -------
        mean: float
            The raw mean of the image in mJy or arbitrary units
        """
        with open( str( path ) ) as file:
            filedata = file.read();
        #include re.DOTALL to make the .*? able to expand over newlines
        exp = re.compile( r"Raw mean \(Stokes I\) =  (\d+\.\d+) mJy" );
        match = exp.search( filedata );
        mean = float( match.group( 1 ) );
        return mean;

    def get_sigma_clipped_mean( self, path: Path ):
        """
        A function to get the sigma clipped mean of a log file at path.

        Parameters
        ----------
        path: Path
            The path to the pybdsf log file

        Returns
        -------
        mean: float
            The sigma clipped mean of the image in mJy or arbitrary units
        """
        with open( str( path ) ) as file:
            filedata = file.read();
        #include re.DOTALL to make the .*? able to expand over newlines
        exp = re.compile( r"sigma clipped mean \(Stokes I\) =  (\d+\.\d+) mJy" );
        match = exp.search( filedata );
        mean = float( match.group( 1 ) );
        return mean;

    def get_rms( self, path: Path ):
        """
        A function to get the rms of a log file at path.

        Parameters
        ----------
        path: Path
            The path to the pybdsf log file

        Returns
        -------
        rms: float
            The raw rms of the image in mJy or arbitrary units
        """
        with open( str( path ) ) as file:
            filedata = file.read();
        exp = re.compile( r"raw rms =  (\d+\.\d+) mJy" );
        match = exp.search( filedata );
        rms = float( match.group( 1 ) );
        return rms;

    def get_sigma_clipped_rms( self, path: Path ):
        """
        A function to get the sigma clipped rms of a log file at path.

        Parameters
        ----------
        path: Path
            The path to the pybdsf log file

        Returns
        -------
        rms: float
            The sigma clipped rms of the image in mJy or arbitrary units
        """
        with open( str( path ) ) as file:
            filedata = file.read();
        exp = re.compile( r"sigma clipped rms =  (\d+\.\d+) mJy" );
        match = exp.search( filedata );
        rms = float( match.group( 1 ) );
        return rms;

    def get_flux_mean_rms( self, path: Path ):
        """
        A function to combine getting the flux, mean, and rms of a log file at path

        Parameters
        ----------
        path: Path
            The path to the pybdsf log file

        Returns
        -------
        flux: float
            The flux of the image in Jy or arbitrary units (because of 0-1 normalizaiton)
        mean: float
            The raw mean of the image in mJy or arbitrary units
        rms: float
            The raw rms of the image in mJy or arbitrary units
        """
        with open( str( path ) ) as file:
            filedata = file.read();
        #include re.DOTALL to make the .*? able to expand over newlines
        exp = re.compile( r"Raw mean \(Stokes I\) =  (\d+\.\d+) mJy and raw rms =  (\d+\.\d+) mJy.*?Flux from sum of \(non-blank\) pixels ..... : (\d+\.\d+) Jy", re.DOTALL );
        match = exp.search( filedata );
        mean = float( match.group( 1 ) );
        rms = float( match.group( 2 ) );
        flux = float( match.group( 3 ) );
        return flux, mean, rms;

    def fmr_data( self, return_nums: bool = False ):
        """
        A function to get a numpy array of shape (len( files ), 3), where the 3 components are flux, mean, and rms in that order.

        WARNING - THIS FUNCTION IS NOT DISTRIBUTABLE - RUN ON A SINGLE NODE ONLY

        Parameters
        ----------
        return_nums : bool = False
            Whether or not to return the numbers captured from the filename

        Returns
        -------
        np.ndarray
            A numpy array of shape (len( files ), 3) containing flux, mean, and rms data for each log file
        np.ndarray (optional)
            An array of the numbers captured from the filenames, only returned if return_nums is True
        """
        if return_nums:
            data, numbers = self.for_each( self.get_flux_mean_rms, self.path / self.subdir, r'.*?image(\d+)\.fits\.pybdsf\.log$', False, None, return_nums );
            return data, numbers;
        else: 
            data = self.for_each( self.get_flux_mean_rms, self.path / self.subdir, r'.*?image(\d+)\.fits\.pybdsf\.log$', False, None, return_nums );
            return data;

    def sigma_clipped_mean_array( self, return_nums: bool = False ):
        """
        A function to get a numpy array of shape (len( files )) of the sigma clipped means

        WARNING - THIS FUNCTION IS NOT DISTRIBUTABLE - RUN ON A SINGLE NODE ONLY

        Parameters
        ----------
        return_nums : bool = False
            Whether or not to return the numbers captured from the filename

        Returns
        -------
        np.ndarray
            A numpy array of shape (len( files )) containing sigma clipped mean data for each log file
        np.ndarray (optional)
            An array of the numbers captured from the filenames, only returned if return_nums is True
        """
        if return_nums:
            data, numbers = self.for_each( self.get_sigma_clipped_mean, self.path / self.subdir, r'.*?image(\d+)\.fits\.pybdsf\.log$', False, None, return_nums );
            return data, numbers;
        else: 
            data = self.for_each( self.get_sigma_clipped_mean, self.path / self.subdir, r'.*?image(\d+)\.fits\.pybdsf\.log$', False, None, return_nums );
            return data;

    def sigma_clipped_rms_array( self, return_nums: bool = False ):
        """
        A function to get a numpy array of shape (len( files )) of the sigma clipped rms's

        WARNING - THIS FUNCTION IS NOT DISTRIBUTABLE - RUN ON A SINGLE NODE ONLY

        Parameters
        ----------
        return_nums : bool = False
            Whether or not to return the numbers captured from the filename

        Returns
        -------
        np.ndarray
            A numpy array of shape (len( files )) containing sigma clipped rms data for each log file
        np.ndarray (optional)
            An array of the numbers captured from the filenames, only returned if return_nums is True
        """
        if return_nums:
            data, numbers = self.for_each( self.get_sigma_clipped_rms, self.path / self.subdir, r'.*?image(\d+)\.fits\.pybdsf\.log$', False, None, return_nums );
            return data, numbers;
        else: 
            data = self.for_each( self.get_sigma_clipped_rms, self.path / self.subdir, r'.*?image(\d+)\.fits\.pybdsf\.log$', False, None, return_nums );
            return data;