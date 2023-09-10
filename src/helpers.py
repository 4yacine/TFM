import copy
import inspect
import logging
import os
import warnings
from typing import List

from src.constants import ALL_ATTR_ACT

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

def merge_list(a,b,override=False):
    """
    function merges a list b into a list a
    
    Args:
        :a (list): first dictionary
        :b (list): second dictionary
        :override (bool): whatever to overide values if the item of b is found in a
        
    Returns:
        :a (list): the merged list
    
    """
    for item in b:
        if item not in a:
            a.append(item)
        elif override != False:
            a[a.index(item)] = item
    return a
    
def sort_modules(dict_config):
    
    new_config = {}
    for key in dict_config:
        if key == "modules":
            new_config[key] = {}
            for mod_order in sorted(dict_config.get(key)):
                new_config[key][mod_order] = dict_config.get(key).get(mod_order)
        else:
            new_config[key] = dict_config.get(key)
    return new_config

def merge_dict(a, b, path=None,override=False,a_dict_name="a",b_dict_name="b"):
    """
    Recursive function merges a dictionary "b" into a dictionary "a"
    
    Args:
        :a (dictionary): first dictionary
        :b (dictionary): second dictionary
        :override (bool): whatever to overide values if the item of b is found in a
        
    Returns:
        :a (dictionary): the merged dictionary
    
    """
    if path is None: path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge_dict(a[key], b[key], path + [str(key)],override,a_dict_name=a_dict_name,b_dict_name=b_dict_name)
            elif isinstance(a[key], list) and isinstance(b[key], list):
                a[key] = merge_list(a[key],b[key], override)
            elif isinstance(a[key], dict) and isinstance(b[key], list):
                item = a[key]
                a[key] = b[key]
                a[key].append(item)
            elif isinstance(a[key], list) and isinstance(b[key], dict):
                a[key].append(b[key])
            elif a[key] == b[key]:
                pass # same leaf value
            else:
                if override != False:
                    log.info(F"Overriding values of {a_dict_name}[{key}] with {b_dict_name}[{key}]")
                    if (inspect.isclass(b[key]) or inspect.isfunction(b[key])):
                        a[key] = b[key]
                    else:
                        a[key] = copy.deepcopy(b[key])
                else:
                    log.debug(F"Conflict at values of {a_dict_name}[{key}]={a[key]} with {b_dict_name}[{key}]={b[key]}")
                    #raise Exception('Conflict at %s' % '.'.join(path + [str(key)]))
        else:
            if (inspect.isclass(b[key]) or inspect.isfunction(b[key]) or inspect.ismodule(b[key])):
                a[key] = b[key]
            elif isinstance(b[key], dict):
                a[key] = {}
                merge_dict(a[key], b[key], path + [str(key)], override, a_dict_name=a_dict_name,
                           b_dict_name=b_dict_name)
            else:
                a[key] = copy.deepcopy(b[key])
    return a


def merge_dict_safe(a, b, path=None,override=False):
    """
    Recursive function merges a dictionary b into a dictionary a.
    Checking for the inputs
    
    Args:
        :a (dictionary): first dictionary
        :b (dictionary): second dictionary
        :override (bool): whatever to override values if the item of b is found in a
        
    Returns:
        :a (dictionary): the merged dictionary
    
    """
    if not isinstance(a, dict):
        if not isinstance(b, dict):
            if a is None or a is False:
                if b is None or b is False:
                    return {}
                else:
                    return {"data":b}
            else:
                if b is None or b is False:
                    return {"data":a}
                else:
                    return {"data_a":a,"data_b":b}
        else:
            if any(b):
                return b
            else:
                return a
    else:
        if not isinstance(b, dict):
            if any(a):
                return a
            else:
                return b
        else:
            return merge_dict(a, b, path, override)
        

def dump(obj):
    """Print all the attributes of an object
    
    Args:
        :obj (obj): the object to print
    """
    for attr in dir(obj):
        print("obj.%s = %r" % (attr, getattr(obj, attr)))
        
def logdump(log_function,obj):
    """log all the attributes of an object
    
    Args:
        :log_function (func): the log function to use
        :obj (obj): the object to print
    """
    log_function("START dump of object [{}]".format(obj))
    for attr in dir(obj):
        log_function("obj.%s = %r" % (attr, getattr(obj, attr)))
    log_function("END dump of object [{}]".format(obj))
    
def rchop(s, sub):
    """
    Remove a substring from the right of a string
    
    Args:
        :s (string): The string to be used
        :sub (string): The substring to be removed
    
    Returns:
        :String: The string result from the chop operation
    """
    return s[:-len(sub)] if s.endswith(sub) else s

def lchop(s, sub):
    """
    Remove a substring from the left of a string
    
    Args:
        :s (string): The string to be used
        :sub (string): The substring to be removed
    
    Returns:
        :String: The string result from the chop operation
    """
    return s[len(sub):] if s.startswith(sub) else s

def remove_xml_namespace_from_list(list_data,xml_elements=[":","@","#"]):
    output_list = [] 
    for item in list_data:
        if isinstance(item, list):
            output_list.append(remove_xml_namespace_from_list(item, xml_elements))
        elif isinstance(item, dict):
            output_list.append(remove_xml_namespace_from_dict(item, xml_elements))
        else:
            output_list.append(item)
    return output_list

def __remove_xml_namespace_from_string(string_key,xml_elements =[":","@","#"] ):
    new_key = string_key
    for item in xml_elements:
        list =new_key.split(item)
        if len(list) > 1:
            new_key = list[1]
    return new_key
def remove_xml_namespace_from_dict(dict_data,xml_elements=[":","@","#"]):
    output_dict = {}
    for key in dict_data:
        key_data = dict_data.get(key)
        new_key = __remove_xml_namespace_from_string(key,xml_elements)
        if isinstance(key_data, list):
            output_dict[new_key] = remove_xml_namespace_from_list(key_data, xml_elements)
        elif isinstance(key_data, dict):
            output_dict[new_key] = remove_xml_namespace_from_dict(key_data, xml_elements)
        else:
            output_dict[new_key] = key_data
    return output_dict



def remove_non_usable_attr(grid2openv, act_attr_to_keep: List[str]) -> List[str]:
    """This function modifies the attribute (of the actions)
    to remove the one that are non usable with your gym environment.

    If only filters things if the default variables are used
    (see _default_act_attr_to_keep)

    Parameters
    ----------
    grid2openv : grid2op.Environment.Environment
        The used grid2op environment
    act_attr_to_keep : List[str]
        The attributes of the actions to keep.

    Returns
    -------
    List[str]
        The same as `act_attr_to_keep` if the user modified the default.
        Or the attributes usable by the environment from the default list.

    """
    modif_attr = act_attr_to_keep
    if act_attr_to_keep == ALL_ATTR_ACT:
        # by default, i remove all the attributes that are not supported by the action type
        # i do not do that if the user specified specific attributes to keep. This is his responsibility in
        # in this case
        modif_attr = []
        for el in act_attr_to_keep:
            if grid2openv.action_space.supports_type(el):
                modif_attr.append(el)
            else:
                warnings.warn(f"attribute {el} cannot be processed by the allowed "
                              "action type. It has been removed from the "
                              "gym space as well.")
    return modif_attr


def create_experiment_gitignore(file_path):
    """
    Function to create a gitignore file inside the experiment output folder to ignore all its contents

    :param file_path: Path of the folder where to create the gitignore file
    :return: None
    """
    # create output dir if not existing
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    with open(os.path.join(file_path,".gitignore"),"w") as f:
        f.write("*")