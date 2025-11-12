# This file has been created by Ashley and Luna. It provides utility functions for distributing the program with SLURM

import os;
import time;
from pathlib import Path;
import shutil;
import numpy as np;
import utils.logging;

# Source - https://stackoverflow.com/a/2785908
# Posted by Alex Martelli, modified by community. See post 'Timeline' for change history
# Retrieved 2025-11-12, License - CC BY-SA 3.0
def wait_until( somepredicate, timeout, period=0.25, *args, **kwargs ):
    mustend = time.time() + timeout
    while time.time() < mustend:
        if somepredicate( *args, **kwargs ): return True
        time.sleep(period)
    return False


class DistributedUtils:
    """
    Utility functions for running on a distributed system (SLURM in particular)
    """
    def __init__( self ):
        self.logger = utils.logging.get_logger( __name__ );

    def is_distributed( self ) -> bool:
        return self.get_task_count() != 1;

    def get_task_id( self ) -> int:
        return int( os.environ.get( "SLURM_ARRAY_TASK_ID", 0 ) );

    def get_task_count( self ) -> int:
        return int( os.environ.get( "SLURM_ARRAY_TASK_COUNT", 1 ) );

    def single_task_only_forcewait( self, taskname: str, function, do_on_array_id: int, *args, **kwargs ):
        """
        A method to execute a function only once when run on a job array, and force all other tasks to wait for completion.

        Wrapping everything you want to do in a single single_task_only_forcewait is sufficient to ensure the tasks
        are performed on only one array element. It is advisable NOT to wrap single_task_only_forcewait calls inside
        other single_task_only_forcewait calls as it will only produce unneccesary overhead

        Paramters
        ---------
        taskname : str
            The name of the task - should be unique and be a valid str to include in an environment variable
        function
            The function to call only once.
        do_on_array_id : int
            The array id on which to call the function. All other array ids must wait for this function
            
        *args, **kwargs
            arguments to pass to the function call
        """
        if do_on_array_id >= self.get_task_count():
            raise ValueError( f'Array id {do_on_array_id} out of range for array of length {self.get_task_count()}' );

        if not self.is_distributed():
            # Doing this separately from if we have multiple array elements allows us to ignore environment variable
            # deletion when this array id == do_on_array_id - hereafter we can assume if self.get_task_id() == do_on_array_id
            # that it is the first array to attempt the problem
            function( *args, **kwargs );
        else:
            if self.get_task_id() == do_on_array_id:
                function( *args, **kwargs );
                os.environ[ f'TASK_{taskname}_COMPLETED' ] = 'True';
            else:
                # Truth value tells us whether or not wait_until timed out
                if not wait_until( lambda : f'TASK_{taskname}_COMPLETED' in os.environ, 10*60 ):
                    raise RuntimeError( f'ERROR - wait_until timed out on array {self.get_task_id()} waiting for {taskname} on array {do_on_array_id}' );

                os.environ[ f'TASK_{taskname}_ARRAY_{self.get_task_id()}_PASS_COMPLETE' ] = 'True';

                # If we're the last array to pass this function, delete the environment variables behind us
                # also we don't need to check for the completed variable since it must exist to get here
                last_array = True;
                for i in range( 0, self.get_task_count() ):
                    if i == do_on_array_id:
                        continue;
                    elif f'TASK_{taskname}_ARRAY_{i}_PASS_COMPLETE' not in os.environ:
                        last_array = False;
                if last_array:
                    os.environ.pop( f'TASK_{taskname}_COMPLETED' );
                    for i in range( 1, self.get_task_count() ):
                        if i == do_on_array_id:
                            continue;
                        else: os.environ.pop( f'TASK_{taskname}_ARRAY_{i}_PASS_COMPLETE' );

    def copy_file_for_multiple_nodes( self, file: Path ):
        """
        Uses a single node (array id 0) to create a copy of a file for each array element,
        with the copy files having a naming scheme according to "[filename]_[id][suffix]" in the same directory.

        For example, on an array of 2 nodes, copying test/test.file would give:
            test/test_0.file
            test/test_1.file
        """
        # Running as a lambda doesn't work for some reason, so just define a function here
        def __copy_file_lambda():
            for i in range( self.get_task_count() ):
                shutil.copyfile( str(file), f'{str(file.parent)}/{file.stem}_{i}{file.suffix}' )
        self.single_task_only_forcewait( f'copy_file_{str(file)}', __copy_file_lambda, 0 );





