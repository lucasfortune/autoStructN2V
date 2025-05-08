# models/factory.py
from .auto_struct_n2v import AutoStructN2VModel

def create_model(stage, **kwargs):
    """
    Factory function to create appropriate model based on the stage.
    
    Args:
        stage (str): Training stage ('stage1' or 'stage2')
        **kwargs: Keyword arguments to pass to the model constructor
        
    Returns:
        nn.Module: Instantiated model
        
    Raises:
        ValueError: If stage is not recognized
    """
    if stage.lower() == 'stage1':
        return AutoStructN2VModel.create_stage1_model(**kwargs)
    elif stage.lower() == 'stage2':
        return AutoStructN2VModel.create_stage2_model(**kwargs)
    else:
        raise ValueError(f"Unknown stage: {stage}. Must be 'stage1' or 'stage2'.")