#!/usr/bin/env python3
import xml.etree.ElementTree as ET
import prepare_isgci_forbidden_fast as m

def parse_full(path):
    classes={}; inclusions=[]
    root=ET.parse(path).getroot()
    for e in root.iter():
        tag=e.tag.split('}')[-1]
        if tag=='GraphClass':
            small=[c.text.strip() for c in e if c.tag.split('}')[-1]=='smallgraph' and c.text]
            classes[e.attrib['id']]=(e.attrib['type'],small)
        elif tag=='incl':
            inclusions.append((e.attrib['sub'],e.attrib['super'],e.attrib.get('confidence')))
    return classes,inclusions

m.parse=parse_full
m.main()
