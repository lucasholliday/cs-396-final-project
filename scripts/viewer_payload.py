"""Compact display JSON; scientific caches and model inputs retain full precision."""
import json,math

def encode_display_data(value):
 def compact(v):
  if isinstance(v,float):return round(v,6) if math.isfinite(v) else None
  if isinstance(v,dict):return {k:compact(x) for k,x in v.items()}
  if isinstance(v,(list,tuple)):return [compact(x) for x in v]
  return v
 return json.dumps(compact(value),allow_nan=False,separators=(',',':')).replace('</','<\\/')
