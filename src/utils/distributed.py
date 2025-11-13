# This file has been created by Ashley and Luna. It provides utility functions for distributing the program with SLURM

import os;
import time;
from pathlib import Path;
import shutil;
import numpy as np;
import utils.logging;
import logging;

# Source - https://stackoverflow.com/a/2785908
# Posted by Alex Martelli, modified by community. See post 'Timeline' for change history
# Retrieved 2025-11-12, License - CC BY-SA 3.0
def wait_until( somepredicate, timeout: int | None, logger, waiting_on_array: int, taskname: str, period=1, *args, **kwargs ):
    mustend = time.time() + timeout if timeout is not None else None
    while ( mustend is None ) or ( time.time() < mustend ):
        logger.debug( f'Waiting on array {waiting_on_array} to complete {taskname}: time={time.time()}' );
        if somepredicate( *args, **kwargs ): 
            logger.debug( f'Array {waiting_on_array} has completed {taskname}: time={time.time()}' );
            return True
        time.sleep(period)
    return False


class DistributedUtils:
    """
    Utility functions for running on a distributed system (SLURM in particular)
    """
    def __init__( self ):
        self.logger = utils.logging.get_logger( __name__, logging.DEBUG );

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
            The name of the task - should be unique and be a valid str to include in a path name
        function
            The function to call only once.
        do_on_array_id : int
            The array id on which to call the function. All other array ids must wait for this function
            
        *args, **kwargs
            arguments to pass to the function call
        """
        if do_on_array_id >= self.get_task_count():
            raise ValueError( f'Array id {do_on_array_id} out of range for array of length {self.get_task_count()}' );
    
        taskname = taskname.replace( '/', '_' ).replace( '-', '_' ).upper();

        if not self.is_distributed():
            # Doing this separately from if we have multiple array elements allows us to ignore flag file
            # deletion when this array id == do_on_array_id - hereafter we can assume if self.get_task_id() == do_on_array_id
            # that it is the first array to attempt the problem
            function( *args, **kwargs );
        else:
            if self.get_task_id() == do_on_array_id:
                function( *args, **kwargs );
                Path( f'TASK_{taskname}_COMPLETED' ).touch();
            else:
                # Truth value tells us whether or not wait_until timed out
                def __is_completed_lambda( taskname ):
                    return Path( f'TASK_{taskname}_COMPLETED' ).exists();
                if not wait_until( __is_completed_lambda, None, self.logger, do_on_array_id, taskname, 1, taskname ):
                    raise RuntimeError( f'ERROR - wait_until timed out on array {self.get_task_id()} waiting for {taskname} on array {do_on_array_id}' );

                Path( f'TASK_{taskname}_ARRAY_{self.get_task_id()}_PASS_COMPLETE' ).touch();

                # If we're the last array to pass this function, delete the environment variables behind us
                # also we don't need to check for the completed variable since it must exist to get here
                last_array = True;
                for i in range( 0, self.get_task_count() ):
                    if i == do_on_array_id:
                        continue;
                    elif not Path( f'TASK_{taskname}_ARRAY_{i}_PASS_COMPLETE' ).exists():
                        last_array = False;
                if last_array:
                    Path( f'TASK_{taskname}_COMPLETED' ).unlink();
                    for i in range( 1, self.get_task_count() ):
                        if i == do_on_array_id:
                            continue;
                        else: Path( f'TASK_{taskname}_ARRAY_{i}_PASS_COMPLETE' ).unlink();
    
    def last_task_only( self, taskname: str, function, *args, **kwargs ):
        """
        Similarly to single_task_only_forcewait this method is used to run a function only once on a single node,
        however this method executes the task on the last node to run. This is useful e.g. if all the tasks are
        doing data preparation and something is to be done once the data is prepared, like graphing the data.

        Paramters
        ---------
        taskname : str
            The name of the task - should be unique and be a valid str to include in a path name
        function
            The function to call only once.
        """
        taskname = taskname.replace( '/', '_' ).replace( '-', '_' ).upper();

        if not self.is_distributed():
            # Doing this separately from if we have multiple array elements allows us to ignore flag files
            function( *args, **kwargs );
        else:
            last_array = True;
            for i in range( 0, self.get_task_count() ):
                if i == self.get_task_id():
                    continue;
                elif not Path( f'TASK_{taskname}_ARRAY_{i}_PASS_COMPLETE' ).exists():
                    Path( f'TASK_{taskname}_ARRAY_{i}_PASS_COMPLETE' ).touch();
                    last_array = False;

            if last_array:
                function( *args, **kwargs );
                for i in range( 1, self.get_task_count() ):
                    if i == self.get_task_id():
                        continue;
                    else: Path( f'TASK_{taskname}_ARRAY_{i}_PASS_COMPLETE' ).unlink();


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





