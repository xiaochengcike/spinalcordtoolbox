#!/usr/bin/env python
# -*- coding: utf-8
# spinal cord toolbox

import sys, os, re


from waflib import TaskGen, Task, Errors, Logs, Utils
from waflib.Configure import conf
from waflib.TaskGen import feature, before_method

def sct_splitext(fn):
	if fn.endswith(".nii.gz"):
		return fn[:-7], ".nii.gz"
	return os.path.splitext(fn)


class sct_task(Task.Task):
	color = "CYAN"

	def keyword(self):
		return ""#self.run_str#self.run_str.split()[0]

	def __str__(self):
		return self.run_str

	def run(self):
		bld = self.generator.bld
		fun, _ = Task.compile_fun(self.run_str, shell=False)
		#print(fun)
		res = fun(self)
		#Task.Task.run(self)
		return res

@TaskGen.feature("sct_propseg")
@TaskGen.before_method('process_source')
def propseg(self):
	#print(self)
	#print(dir(self))

	cwd = getattr(self, "cwd", None)
	bld = self.bld

	if cwd is None:
		cwd = bld.path.get_bld()

	if isinstance(cwd, str):
		cwd = bld.path.find_dir(cwd)

	src = self.to_nodes(self.input)
	assert len(src) == 1

	base, ext = sct_splitext(src[0].name)
	output_name = "{}_seg{}".format(base, ext)
	output = cwd.find_or_declare(output_name)
	tgt = [output]

	task = self.create_task('sct_task', src, tgt,
	 run_str="sct_propseg -i %s -c %s" % (src[0].abspath(), self.contrast),
	 cwd=cwd,
	)

	self.segmented = output

	#self.source = []
	#self.target = tgt


@conf
def sct_propseg(bld, **kw):
	res = bld(
	 features = "sct_propseg",
	 **kw
	)
	res.post()
	return res



@TaskGen.feature("sct_label_vertebrae")
@TaskGen.before_method('process_source')
def label_vertebrae(self):

	cwd = getattr(self, "cwd", None)
	bld = self.bld

	if cwd is None:
		cwd = bld.path.get_bld()

	if isinstance(cwd, str):
		cwd = bld.path.find_dir(cwd)

	src = self.to_nodes(self.input) + self.to_nodes(self.segmented)
	assert len(src) == 2

	base, ext = sct_splitext(src[0].name)
	output_name = "{}_seg_labeled{}".format(base, ext)
	output = cwd.find_or_declare(output_name)
	tgt = [output]

	task = self.create_task('sct_task', src, tgt,
	 run_str="sct_label_vertebrae -i %s -s %s -c %s" \
	  % (src[0].abspath(), src[1].abspath(), self.contrast),
	 cwd=cwd,
	)

	#self.source = []
	#self.target = tgt

	self.labeled = output

@conf
def sct_label_vertebrae(bld, **kw):
	res = bld(
	 features = "sct_label_vertebrae",
	 **kw
	)
	res.post()
	return res



@TaskGen.feature("sct_label_utils")
@TaskGen.before_method('process_source')
def label_utils(self):

	cwd = getattr(self, "cwd", None)
	bld = self.bld

	if cwd is None:
		cwd = bld.path.get_bld()

	if isinstance(cwd, str):
		cwd = bld.path.find_dir(cwd)

	src = self.to_nodes(self.input)
	assert len(src) == 1

	base, ext = sct_splitext(src[0].name)
	output_name = "labels{}".format(ext)
	output = cwd.find_or_declare(output_name)
	tgt = [output]

	vert_body = getattr(self, "vert_body", None)
	if vert_body:
		vert_body = "-vert-body {}".format(",".join([str(x) for x in vert_body]))
	else:
		vert_body = ""

	task = self.create_task('sct_task', src, tgt,
	 run_str="sct_label_utils -i %s %s" \
	  % (src[0].abspath(), vert_body),
	 cwd=cwd,
	)

	self.labels = output

	#self.source = []
	#self.target = tgt

@conf
def sct_label_utils(bld, **kw):
	res = bld(
	 features = "sct_label_utils",
	 **kw
	)

	res.post()

	return res



@TaskGen.feature("sct_register_to_template")
@TaskGen.before_method('process_source')
def register_to_template(self):

	cwd = getattr(self, "cwd", None)
	bld = self.bld

	if cwd is None:
		cwd = bld.path.get_bld()

	if isinstance(cwd, str):
		cwd = bld.path.find_dir(cwd)

	src = self.to_nodes(self.input) + self.to_nodes(self.segmented) + self.to_nodes(self.labels)
	assert len(src) == 3

	base, ext = sct_splitext(src[0].name)
	self.a2t = cwd.find_or_declare("warp_anat2template{}".format(ext))
	self.t2a = cwd.find_or_declare("warp_template2anat{}".format(ext))
	self.c2s = cwd.find_or_declare("warp_curve2straight{}".format(ext))
	self.s2c = cwd.find_or_declare("warp_straight2curve{}".format(ext))

	tgt = [
	 self.a2t,
	 self.t2a,
	 self.c2s,
	 self.s2c,
	]

	task = self.create_task('sct_task', src, tgt,
	 run_str="sct_register_to_template -i %s -s %s -l %s -c %s" \
	  % (src[0].abspath(), src[1].abspath(), src[2].abspath(), self.contrast),
	 cwd=cwd,
	)


@conf
def sct_register_to_template(bld, **kw):
	res = bld(
	 features = "sct_register_to_template",
	 **kw
	)
	res.post()
	return res


def configure(self):
	self.find_program('sct_check_dependencies', var='sct_check_dependencies', mandatory=False)
	self.find_program('sct_propseg', var='sct_propseg', mandatory=False)
