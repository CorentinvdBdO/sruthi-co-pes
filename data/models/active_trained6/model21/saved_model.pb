??	
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
:
Maximum
x"T
y"T
z"T"
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
-
Sqrt
x"T
y"T"
Ttype:

2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
;
Sub
x"T
y"T
z"T"
Ttype:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??
`
meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namemean
Y
mean/Read/ReadVariableOpReadVariableOpmean*
_output_shapes
:*
dtype0
h
varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
variance
a
variance/Read/ReadVariableOpReadVariableOpvariance*
_output_shapes
:*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0	
}
dense_105/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*!
shared_namedense_105/kernel
v
$dense_105/kernel/Read/ReadVariableOpReadVariableOpdense_105/kernel*
_output_shapes
:	?*
dtype0
u
dense_105/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_105/bias
n
"dense_105/bias/Read/ReadVariableOpReadVariableOpdense_105/bias*
_output_shapes	
:?*
dtype0
~
dense_106/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*!
shared_namedense_106/kernel
w
$dense_106/kernel/Read/ReadVariableOpReadVariableOpdense_106/kernel* 
_output_shapes
:
??*
dtype0
u
dense_106/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_106/bias
n
"dense_106/bias/Read/ReadVariableOpReadVariableOpdense_106/bias*
_output_shapes	
:?*
dtype0
~
dense_107/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*!
shared_namedense_107/kernel
w
$dense_107/kernel/Read/ReadVariableOpReadVariableOpdense_107/kernel* 
_output_shapes
:
??*
dtype0
u
dense_107/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_107/bias
n
"dense_107/bias/Read/ReadVariableOpReadVariableOpdense_107/bias*
_output_shapes	
:?*
dtype0
~
dense_108/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*!
shared_namedense_108/kernel
w
$dense_108/kernel/Read/ReadVariableOpReadVariableOpdense_108/kernel* 
_output_shapes
:
??*
dtype0
u
dense_108/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_108/bias
n
"dense_108/bias/Read/ReadVariableOpReadVariableOpdense_108/bias*
_output_shapes	
:?*
dtype0
}
dense_109/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*!
shared_namedense_109/kernel
v
$dense_109/kernel/Read/ReadVariableOpReadVariableOpdense_109/kernel*
_output_shapes
:	?*
dtype0
t
dense_109/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_109/bias
m
"dense_109/bias/Read/ReadVariableOpReadVariableOpdense_109/bias*
_output_shapes
:*
dtype0
j
Adamax/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameAdamax/iter
c
Adamax/iter/Read/ReadVariableOpReadVariableOpAdamax/iter*
_output_shapes
: *
dtype0	
n
Adamax/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdamax/beta_1
g
!Adamax/beta_1/Read/ReadVariableOpReadVariableOpAdamax/beta_1*
_output_shapes
: *
dtype0
n
Adamax/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdamax/beta_2
g
!Adamax/beta_2/Read/ReadVariableOpReadVariableOpAdamax/beta_2*
_output_shapes
: *
dtype0
l
Adamax/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdamax/decay
e
 Adamax/decay/Read/ReadVariableOpReadVariableOpAdamax/decay*
_output_shapes
: *
dtype0
|
Adamax/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdamax/learning_rate
u
(Adamax/learning_rate/Read/ReadVariableOpReadVariableOpAdamax/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adamax/dense_105/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?**
shared_nameAdamax/dense_105/kernel/m
?
-Adamax/dense_105/kernel/m/Read/ReadVariableOpReadVariableOpAdamax/dense_105/kernel/m*
_output_shapes
:	?*
dtype0
?
Adamax/dense_105/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*(
shared_nameAdamax/dense_105/bias/m
?
+Adamax/dense_105/bias/m/Read/ReadVariableOpReadVariableOpAdamax/dense_105/bias/m*
_output_shapes	
:?*
dtype0
?
Adamax/dense_106/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??**
shared_nameAdamax/dense_106/kernel/m
?
-Adamax/dense_106/kernel/m/Read/ReadVariableOpReadVariableOpAdamax/dense_106/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adamax/dense_106/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*(
shared_nameAdamax/dense_106/bias/m
?
+Adamax/dense_106/bias/m/Read/ReadVariableOpReadVariableOpAdamax/dense_106/bias/m*
_output_shapes	
:?*
dtype0
?
Adamax/dense_107/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??**
shared_nameAdamax/dense_107/kernel/m
?
-Adamax/dense_107/kernel/m/Read/ReadVariableOpReadVariableOpAdamax/dense_107/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adamax/dense_107/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*(
shared_nameAdamax/dense_107/bias/m
?
+Adamax/dense_107/bias/m/Read/ReadVariableOpReadVariableOpAdamax/dense_107/bias/m*
_output_shapes	
:?*
dtype0
?
Adamax/dense_108/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??**
shared_nameAdamax/dense_108/kernel/m
?
-Adamax/dense_108/kernel/m/Read/ReadVariableOpReadVariableOpAdamax/dense_108/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adamax/dense_108/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*(
shared_nameAdamax/dense_108/bias/m
?
+Adamax/dense_108/bias/m/Read/ReadVariableOpReadVariableOpAdamax/dense_108/bias/m*
_output_shapes	
:?*
dtype0
?
Adamax/dense_109/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?**
shared_nameAdamax/dense_109/kernel/m
?
-Adamax/dense_109/kernel/m/Read/ReadVariableOpReadVariableOpAdamax/dense_109/kernel/m*
_output_shapes
:	?*
dtype0
?
Adamax/dense_109/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdamax/dense_109/bias/m

+Adamax/dense_109/bias/m/Read/ReadVariableOpReadVariableOpAdamax/dense_109/bias/m*
_output_shapes
:*
dtype0
?
Adamax/dense_105/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?**
shared_nameAdamax/dense_105/kernel/v
?
-Adamax/dense_105/kernel/v/Read/ReadVariableOpReadVariableOpAdamax/dense_105/kernel/v*
_output_shapes
:	?*
dtype0
?
Adamax/dense_105/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*(
shared_nameAdamax/dense_105/bias/v
?
+Adamax/dense_105/bias/v/Read/ReadVariableOpReadVariableOpAdamax/dense_105/bias/v*
_output_shapes	
:?*
dtype0
?
Adamax/dense_106/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??**
shared_nameAdamax/dense_106/kernel/v
?
-Adamax/dense_106/kernel/v/Read/ReadVariableOpReadVariableOpAdamax/dense_106/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adamax/dense_106/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*(
shared_nameAdamax/dense_106/bias/v
?
+Adamax/dense_106/bias/v/Read/ReadVariableOpReadVariableOpAdamax/dense_106/bias/v*
_output_shapes	
:?*
dtype0
?
Adamax/dense_107/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??**
shared_nameAdamax/dense_107/kernel/v
?
-Adamax/dense_107/kernel/v/Read/ReadVariableOpReadVariableOpAdamax/dense_107/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adamax/dense_107/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*(
shared_nameAdamax/dense_107/bias/v
?
+Adamax/dense_107/bias/v/Read/ReadVariableOpReadVariableOpAdamax/dense_107/bias/v*
_output_shapes	
:?*
dtype0
?
Adamax/dense_108/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??**
shared_nameAdamax/dense_108/kernel/v
?
-Adamax/dense_108/kernel/v/Read/ReadVariableOpReadVariableOpAdamax/dense_108/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adamax/dense_108/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*(
shared_nameAdamax/dense_108/bias/v
?
+Adamax/dense_108/bias/v/Read/ReadVariableOpReadVariableOpAdamax/dense_108/bias/v*
_output_shapes	
:?*
dtype0
?
Adamax/dense_109/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?**
shared_nameAdamax/dense_109/kernel/v
?
-Adamax/dense_109/kernel/v/Read/ReadVariableOpReadVariableOpAdamax/dense_109/kernel/v*
_output_shapes
:	?*
dtype0
?
Adamax/dense_109/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdamax/dense_109/bias/v

+Adamax/dense_109/bias/v/Read/ReadVariableOpReadVariableOpAdamax/dense_109/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?9
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?8
value?8B?8 B?8
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
layer_with_weights-5
layer-5
	optimizer
regularization_losses
	trainable_variables

	variables
	keras_api

signatures
]
state_variables
_broadcast_shape
mean
variance
	count
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
 bias
!regularization_losses
"trainable_variables
#	variables
$	keras_api
h

%kernel
&bias
'regularization_losses
(trainable_variables
)	variables
*	keras_api
h

+kernel
,bias
-regularization_losses
.trainable_variables
/	variables
0	keras_api
?
1iter

2beta_1

3beta_2
	4decay
5learning_ratemYmZm[m\m] m^%m_&m`+ma,mbvcvdvevfvg vh%vi&vj+vk,vl
 
F
0
1
2
3
4
 5
%6
&7
+8
,9
^
0
1
2
3
4
5
6
7
 8
%9
&10
+11
,12
?
6layer_metrics
7non_trainable_variables
8metrics
9layer_regularization_losses
regularization_losses
	trainable_variables

	variables

:layers
 
#
mean
variance
	count
 
NL
VARIABLE_VALUEmean4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEvariance8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEcount5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUE
 
\Z
VARIABLE_VALUEdense_105/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_105/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
;layer_metrics
<non_trainable_variables
=metrics
>layer_regularization_losses
regularization_losses
trainable_variables
	variables

?layers
\Z
VARIABLE_VALUEdense_106/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_106/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
@layer_metrics
Anon_trainable_variables
Bmetrics
Clayer_regularization_losses
regularization_losses
trainable_variables
	variables

Dlayers
\Z
VARIABLE_VALUEdense_107/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_107/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
 1

0
 1
?
Elayer_metrics
Fnon_trainable_variables
Gmetrics
Hlayer_regularization_losses
!regularization_losses
"trainable_variables
#	variables

Ilayers
\Z
VARIABLE_VALUEdense_108/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_108/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

%0
&1

%0
&1
?
Jlayer_metrics
Knon_trainable_variables
Lmetrics
Mlayer_regularization_losses
'regularization_losses
(trainable_variables
)	variables

Nlayers
\Z
VARIABLE_VALUEdense_109/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_109/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

+0
,1

+0
,1
?
Olayer_metrics
Pnon_trainable_variables
Qmetrics
Rlayer_regularization_losses
-regularization_losses
.trainable_variables
/	variables

Slayers
JH
VARIABLE_VALUEAdamax/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEAdamax/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEAdamax/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdamax/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEAdamax/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
2

T0
 
*
0
1
2
3
4
5
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	Utotal
	Vcount
W	variables
X	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

U0
V1

W	variables
?
VARIABLE_VALUEAdamax/dense_105/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdamax/dense_105/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdamax/dense_106/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdamax/dense_106/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdamax/dense_107/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdamax/dense_107/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdamax/dense_108/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdamax/dense_108/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdamax/dense_109/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdamax/dense_109/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdamax/dense_105/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdamax/dense_105/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdamax/dense_106/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdamax/dense_106/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdamax/dense_107/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdamax/dense_107/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdamax/dense_108/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdamax/dense_108/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdamax/dense_109/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdamax/dense_109/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
#serving_default_normalization_inputPlaceholder*0
_output_shapes
:??????????????????*
dtype0*%
shape:??????????????????
?
StatefulPartitionedCallStatefulPartitionedCall#serving_default_normalization_inputmeanvariancedense_105/kerneldense_105/biasdense_106/kerneldense_106/biasdense_107/kerneldense_107/biasdense_108/kerneldense_108/biasdense_109/kerneldense_109/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *.
f)R'
%__inference_signature_wrapper_2763250
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamemean/Read/ReadVariableOpvariance/Read/ReadVariableOpcount/Read/ReadVariableOp$dense_105/kernel/Read/ReadVariableOp"dense_105/bias/Read/ReadVariableOp$dense_106/kernel/Read/ReadVariableOp"dense_106/bias/Read/ReadVariableOp$dense_107/kernel/Read/ReadVariableOp"dense_107/bias/Read/ReadVariableOp$dense_108/kernel/Read/ReadVariableOp"dense_108/bias/Read/ReadVariableOp$dense_109/kernel/Read/ReadVariableOp"dense_109/bias/Read/ReadVariableOpAdamax/iter/Read/ReadVariableOp!Adamax/beta_1/Read/ReadVariableOp!Adamax/beta_2/Read/ReadVariableOp Adamax/decay/Read/ReadVariableOp(Adamax/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount_1/Read/ReadVariableOp-Adamax/dense_105/kernel/m/Read/ReadVariableOp+Adamax/dense_105/bias/m/Read/ReadVariableOp-Adamax/dense_106/kernel/m/Read/ReadVariableOp+Adamax/dense_106/bias/m/Read/ReadVariableOp-Adamax/dense_107/kernel/m/Read/ReadVariableOp+Adamax/dense_107/bias/m/Read/ReadVariableOp-Adamax/dense_108/kernel/m/Read/ReadVariableOp+Adamax/dense_108/bias/m/Read/ReadVariableOp-Adamax/dense_109/kernel/m/Read/ReadVariableOp+Adamax/dense_109/bias/m/Read/ReadVariableOp-Adamax/dense_105/kernel/v/Read/ReadVariableOp+Adamax/dense_105/bias/v/Read/ReadVariableOp-Adamax/dense_106/kernel/v/Read/ReadVariableOp+Adamax/dense_106/bias/v/Read/ReadVariableOp-Adamax/dense_107/kernel/v/Read/ReadVariableOp+Adamax/dense_107/bias/v/Read/ReadVariableOp-Adamax/dense_108/kernel/v/Read/ReadVariableOp+Adamax/dense_108/bias/v/Read/ReadVariableOp-Adamax/dense_109/kernel/v/Read/ReadVariableOp+Adamax/dense_109/bias/v/Read/ReadVariableOpConst*5
Tin.
,2*		*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__traced_save_2763652
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamemeanvariancecountdense_105/kerneldense_105/biasdense_106/kerneldense_106/biasdense_107/kerneldense_107/biasdense_108/kerneldense_108/biasdense_109/kerneldense_109/biasAdamax/iterAdamax/beta_1Adamax/beta_2Adamax/decayAdamax/learning_ratetotalcount_1Adamax/dense_105/kernel/mAdamax/dense_105/bias/mAdamax/dense_106/kernel/mAdamax/dense_106/bias/mAdamax/dense_107/kernel/mAdamax/dense_107/bias/mAdamax/dense_108/kernel/mAdamax/dense_108/bias/mAdamax/dense_109/kernel/mAdamax/dense_109/bias/mAdamax/dense_105/kernel/vAdamax/dense_105/bias/vAdamax/dense_106/kernel/vAdamax/dense_106/bias/vAdamax/dense_107/kernel/vAdamax/dense_107/bias/vAdamax/dense_108/kernel/vAdamax/dense_108/bias/vAdamax/dense_109/kernel/vAdamax/dense_109/bias/v*4
Tin-
+2)*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference__traced_restore_2763782??
?	
?
F__inference_dense_109_layer_call_and_return_conditional_losses_2763009

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
F__inference_dense_107_layer_call_and_return_conditional_losses_2763461

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
+__inference_dense_107_layer_call_fn_2763470

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_107_layer_call_and_return_conditional_losses_27629562
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
F__inference_dense_109_layer_call_and_return_conditional_losses_2763500

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?A
?
J__inference_sequential_21_layer_call_and_return_conditional_losses_2763301

inputs1
-normalization_reshape_readvariableop_resource3
/normalization_reshape_1_readvariableop_resource,
(dense_105_matmul_readvariableop_resource-
)dense_105_biasadd_readvariableop_resource,
(dense_106_matmul_readvariableop_resource-
)dense_106_biasadd_readvariableop_resource,
(dense_107_matmul_readvariableop_resource-
)dense_107_biasadd_readvariableop_resource,
(dense_108_matmul_readvariableop_resource-
)dense_108_biasadd_readvariableop_resource,
(dense_109_matmul_readvariableop_resource-
)dense_109_biasadd_readvariableop_resource
identity?? dense_105/BiasAdd/ReadVariableOp?dense_105/MatMul/ReadVariableOp? dense_106/BiasAdd/ReadVariableOp?dense_106/MatMul/ReadVariableOp? dense_107/BiasAdd/ReadVariableOp?dense_107/MatMul/ReadVariableOp? dense_108/BiasAdd/ReadVariableOp?dense_108/MatMul/ReadVariableOp? dense_109/BiasAdd/ReadVariableOp?dense_109/MatMul/ReadVariableOp?$normalization/Reshape/ReadVariableOp?&normalization/Reshape_1/ReadVariableOp?
$normalization/Reshape/ReadVariableOpReadVariableOp-normalization_reshape_readvariableop_resource*
_output_shapes
:*
dtype02&
$normalization/Reshape/ReadVariableOp?
normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization/Reshape/shape?
normalization/ReshapeReshape,normalization/Reshape/ReadVariableOp:value:0$normalization/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization/Reshape?
&normalization/Reshape_1/ReadVariableOpReadVariableOp/normalization_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02(
&normalization/Reshape_1/ReadVariableOp?
normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization/Reshape_1/shape?
normalization/Reshape_1Reshape.normalization/Reshape_1/ReadVariableOp:value:0&normalization/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization/Reshape_1?
normalization/subSubinputsnormalization/Reshape:output:0*
T0*'
_output_shapes
:?????????2
normalization/sub{
normalization/SqrtSqrt normalization/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization/Sqrtw
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization/Maximum/y?
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization/Maximum?
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization/truediv?
dense_105/MatMul/ReadVariableOpReadVariableOp(dense_105_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02!
dense_105/MatMul/ReadVariableOp?
dense_105/MatMulMatMulnormalization/truediv:z:0'dense_105/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_105/MatMul?
 dense_105/BiasAdd/ReadVariableOpReadVariableOp)dense_105_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_105/BiasAdd/ReadVariableOp?
dense_105/BiasAddBiasAdddense_105/MatMul:product:0(dense_105/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_105/BiasAddw
dense_105/ReluReludense_105/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_105/Relu?
dense_106/MatMul/ReadVariableOpReadVariableOp(dense_106_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02!
dense_106/MatMul/ReadVariableOp?
dense_106/MatMulMatMuldense_105/Relu:activations:0'dense_106/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_106/MatMul?
 dense_106/BiasAdd/ReadVariableOpReadVariableOp)dense_106_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_106/BiasAdd/ReadVariableOp?
dense_106/BiasAddBiasAdddense_106/MatMul:product:0(dense_106/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_106/BiasAddw
dense_106/ReluReludense_106/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_106/Relu?
dense_107/MatMul/ReadVariableOpReadVariableOp(dense_107_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02!
dense_107/MatMul/ReadVariableOp?
dense_107/MatMulMatMuldense_106/Relu:activations:0'dense_107/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_107/MatMul?
 dense_107/BiasAdd/ReadVariableOpReadVariableOp)dense_107_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_107/BiasAdd/ReadVariableOp?
dense_107/BiasAddBiasAdddense_107/MatMul:product:0(dense_107/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_107/BiasAddw
dense_107/ReluReludense_107/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_107/Relu?
dense_108/MatMul/ReadVariableOpReadVariableOp(dense_108_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02!
dense_108/MatMul/ReadVariableOp?
dense_108/MatMulMatMuldense_107/Relu:activations:0'dense_108/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_108/MatMul?
 dense_108/BiasAdd/ReadVariableOpReadVariableOp)dense_108_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_108/BiasAdd/ReadVariableOp?
dense_108/BiasAddBiasAdddense_108/MatMul:product:0(dense_108/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_108/BiasAddw
dense_108/ReluReludense_108/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_108/Relu?
dense_109/MatMul/ReadVariableOpReadVariableOp(dense_109_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02!
dense_109/MatMul/ReadVariableOp?
dense_109/MatMulMatMuldense_108/Relu:activations:0'dense_109/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_109/MatMul?
 dense_109/BiasAdd/ReadVariableOpReadVariableOp)dense_109_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_109/BiasAdd/ReadVariableOp?
dense_109/BiasAddBiasAdddense_109/MatMul:product:0(dense_109/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_109/BiasAdd?
IdentityIdentitydense_109/BiasAdd:output:0!^dense_105/BiasAdd/ReadVariableOp ^dense_105/MatMul/ReadVariableOp!^dense_106/BiasAdd/ReadVariableOp ^dense_106/MatMul/ReadVariableOp!^dense_107/BiasAdd/ReadVariableOp ^dense_107/MatMul/ReadVariableOp!^dense_108/BiasAdd/ReadVariableOp ^dense_108/MatMul/ReadVariableOp!^dense_109/BiasAdd/ReadVariableOp ^dense_109/MatMul/ReadVariableOp%^normalization/Reshape/ReadVariableOp'^normalization/Reshape_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:??????????????????::::::::::::2D
 dense_105/BiasAdd/ReadVariableOp dense_105/BiasAdd/ReadVariableOp2B
dense_105/MatMul/ReadVariableOpdense_105/MatMul/ReadVariableOp2D
 dense_106/BiasAdd/ReadVariableOp dense_106/BiasAdd/ReadVariableOp2B
dense_106/MatMul/ReadVariableOpdense_106/MatMul/ReadVariableOp2D
 dense_107/BiasAdd/ReadVariableOp dense_107/BiasAdd/ReadVariableOp2B
dense_107/MatMul/ReadVariableOpdense_107/MatMul/ReadVariableOp2D
 dense_108/BiasAdd/ReadVariableOp dense_108/BiasAdd/ReadVariableOp2B
dense_108/MatMul/ReadVariableOpdense_108/MatMul/ReadVariableOp2D
 dense_109/BiasAdd/ReadVariableOp dense_109/BiasAdd/ReadVariableOp2B
dense_109/MatMul/ReadVariableOpdense_109/MatMul/ReadVariableOp2L
$normalization/Reshape/ReadVariableOp$normalization/Reshape/ReadVariableOp2P
&normalization/Reshape_1/ReadVariableOp&normalization/Reshape_1/ReadVariableOp:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?
?
+__inference_dense_108_layer_call_fn_2763490

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_108_layer_call_and_return_conditional_losses_27629832
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?-
?
J__inference_sequential_21_layer_call_and_return_conditional_losses_2763184

inputs1
-normalization_reshape_readvariableop_resource3
/normalization_reshape_1_readvariableop_resource
dense_105_2763158
dense_105_2763160
dense_106_2763163
dense_106_2763165
dense_107_2763168
dense_107_2763170
dense_108_2763173
dense_108_2763175
dense_109_2763178
dense_109_2763180
identity??!dense_105/StatefulPartitionedCall?!dense_106/StatefulPartitionedCall?!dense_107/StatefulPartitionedCall?!dense_108/StatefulPartitionedCall?!dense_109/StatefulPartitionedCall?$normalization/Reshape/ReadVariableOp?&normalization/Reshape_1/ReadVariableOp?
$normalization/Reshape/ReadVariableOpReadVariableOp-normalization_reshape_readvariableop_resource*
_output_shapes
:*
dtype02&
$normalization/Reshape/ReadVariableOp?
normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization/Reshape/shape?
normalization/ReshapeReshape,normalization/Reshape/ReadVariableOp:value:0$normalization/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization/Reshape?
&normalization/Reshape_1/ReadVariableOpReadVariableOp/normalization_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02(
&normalization/Reshape_1/ReadVariableOp?
normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization/Reshape_1/shape?
normalization/Reshape_1Reshape.normalization/Reshape_1/ReadVariableOp:value:0&normalization/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization/Reshape_1?
normalization/subSubinputsnormalization/Reshape:output:0*
T0*'
_output_shapes
:?????????2
normalization/sub{
normalization/SqrtSqrt normalization/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization/Sqrtw
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization/Maximum/y?
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization/Maximum?
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization/truediv?
!dense_105/StatefulPartitionedCallStatefulPartitionedCallnormalization/truediv:z:0dense_105_2763158dense_105_2763160*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_105_layer_call_and_return_conditional_losses_27629022#
!dense_105/StatefulPartitionedCall?
!dense_106/StatefulPartitionedCallStatefulPartitionedCall*dense_105/StatefulPartitionedCall:output:0dense_106_2763163dense_106_2763165*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_106_layer_call_and_return_conditional_losses_27629292#
!dense_106/StatefulPartitionedCall?
!dense_107/StatefulPartitionedCallStatefulPartitionedCall*dense_106/StatefulPartitionedCall:output:0dense_107_2763168dense_107_2763170*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_107_layer_call_and_return_conditional_losses_27629562#
!dense_107/StatefulPartitionedCall?
!dense_108/StatefulPartitionedCallStatefulPartitionedCall*dense_107/StatefulPartitionedCall:output:0dense_108_2763173dense_108_2763175*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_108_layer_call_and_return_conditional_losses_27629832#
!dense_108/StatefulPartitionedCall?
!dense_109/StatefulPartitionedCallStatefulPartitionedCall*dense_108/StatefulPartitionedCall:output:0dense_109_2763178dense_109_2763180*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_109_layer_call_and_return_conditional_losses_27630092#
!dense_109/StatefulPartitionedCall?
IdentityIdentity*dense_109/StatefulPartitionedCall:output:0"^dense_105/StatefulPartitionedCall"^dense_106/StatefulPartitionedCall"^dense_107/StatefulPartitionedCall"^dense_108/StatefulPartitionedCall"^dense_109/StatefulPartitionedCall%^normalization/Reshape/ReadVariableOp'^normalization/Reshape_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:??????????????????::::::::::::2F
!dense_105/StatefulPartitionedCall!dense_105/StatefulPartitionedCall2F
!dense_106/StatefulPartitionedCall!dense_106/StatefulPartitionedCall2F
!dense_107/StatefulPartitionedCall!dense_107/StatefulPartitionedCall2F
!dense_108/StatefulPartitionedCall!dense_108/StatefulPartitionedCall2F
!dense_109/StatefulPartitionedCall!dense_109/StatefulPartitionedCall2L
$normalization/Reshape/ReadVariableOp$normalization/Reshape/ReadVariableOp2P
&normalization/Reshape_1/ReadVariableOp&normalization/Reshape_1/ReadVariableOp:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?U
?
 __inference__traced_save_2763652
file_prefix#
savev2_mean_read_readvariableop'
#savev2_variance_read_readvariableop$
 savev2_count_read_readvariableop	/
+savev2_dense_105_kernel_read_readvariableop-
)savev2_dense_105_bias_read_readvariableop/
+savev2_dense_106_kernel_read_readvariableop-
)savev2_dense_106_bias_read_readvariableop/
+savev2_dense_107_kernel_read_readvariableop-
)savev2_dense_107_bias_read_readvariableop/
+savev2_dense_108_kernel_read_readvariableop-
)savev2_dense_108_bias_read_readvariableop/
+savev2_dense_109_kernel_read_readvariableop-
)savev2_dense_109_bias_read_readvariableop*
&savev2_adamax_iter_read_readvariableop	,
(savev2_adamax_beta_1_read_readvariableop,
(savev2_adamax_beta_2_read_readvariableop+
'savev2_adamax_decay_read_readvariableop3
/savev2_adamax_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop&
"savev2_count_1_read_readvariableop8
4savev2_adamax_dense_105_kernel_m_read_readvariableop6
2savev2_adamax_dense_105_bias_m_read_readvariableop8
4savev2_adamax_dense_106_kernel_m_read_readvariableop6
2savev2_adamax_dense_106_bias_m_read_readvariableop8
4savev2_adamax_dense_107_kernel_m_read_readvariableop6
2savev2_adamax_dense_107_bias_m_read_readvariableop8
4savev2_adamax_dense_108_kernel_m_read_readvariableop6
2savev2_adamax_dense_108_bias_m_read_readvariableop8
4savev2_adamax_dense_109_kernel_m_read_readvariableop6
2savev2_adamax_dense_109_bias_m_read_readvariableop8
4savev2_adamax_dense_105_kernel_v_read_readvariableop6
2savev2_adamax_dense_105_bias_v_read_readvariableop8
4savev2_adamax_dense_106_kernel_v_read_readvariableop6
2savev2_adamax_dense_106_bias_v_read_readvariableop8
4savev2_adamax_dense_107_kernel_v_read_readvariableop6
2savev2_adamax_dense_107_bias_v_read_readvariableop8
4savev2_adamax_dense_108_kernel_v_read_readvariableop6
2savev2_adamax_dense_108_bias_v_read_readvariableop8
4savev2_adamax_dense_109_kernel_v_read_readvariableop6
2savev2_adamax_dense_109_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:)*
dtype0*?
value?B?)B4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:)*
dtype0*e
value\BZ)B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0savev2_mean_read_readvariableop#savev2_variance_read_readvariableop savev2_count_read_readvariableop+savev2_dense_105_kernel_read_readvariableop)savev2_dense_105_bias_read_readvariableop+savev2_dense_106_kernel_read_readvariableop)savev2_dense_106_bias_read_readvariableop+savev2_dense_107_kernel_read_readvariableop)savev2_dense_107_bias_read_readvariableop+savev2_dense_108_kernel_read_readvariableop)savev2_dense_108_bias_read_readvariableop+savev2_dense_109_kernel_read_readvariableop)savev2_dense_109_bias_read_readvariableop&savev2_adamax_iter_read_readvariableop(savev2_adamax_beta_1_read_readvariableop(savev2_adamax_beta_2_read_readvariableop'savev2_adamax_decay_read_readvariableop/savev2_adamax_learning_rate_read_readvariableop savev2_total_read_readvariableop"savev2_count_1_read_readvariableop4savev2_adamax_dense_105_kernel_m_read_readvariableop2savev2_adamax_dense_105_bias_m_read_readvariableop4savev2_adamax_dense_106_kernel_m_read_readvariableop2savev2_adamax_dense_106_bias_m_read_readvariableop4savev2_adamax_dense_107_kernel_m_read_readvariableop2savev2_adamax_dense_107_bias_m_read_readvariableop4savev2_adamax_dense_108_kernel_m_read_readvariableop2savev2_adamax_dense_108_bias_m_read_readvariableop4savev2_adamax_dense_109_kernel_m_read_readvariableop2savev2_adamax_dense_109_bias_m_read_readvariableop4savev2_adamax_dense_105_kernel_v_read_readvariableop2savev2_adamax_dense_105_bias_v_read_readvariableop4savev2_adamax_dense_106_kernel_v_read_readvariableop2savev2_adamax_dense_106_bias_v_read_readvariableop4savev2_adamax_dense_107_kernel_v_read_readvariableop2savev2_adamax_dense_107_bias_v_read_readvariableop4savev2_adamax_dense_108_kernel_v_read_readvariableop2savev2_adamax_dense_108_bias_v_read_readvariableop4savev2_adamax_dense_109_kernel_v_read_readvariableop2savev2_adamax_dense_109_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *7
dtypes-
+2)		2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: ::: :	?:?:
??:?:
??:?:
??:?:	?:: : : : : : : :	?:?:
??:?:
??:?:
??:?:	?::	?:?:
??:?:
??:?:
??:?:	?:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :%!

_output_shapes
:	?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!	

_output_shapes	
:?:&
"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	?:! 

_output_shapes	
:?:&!"
 
_output_shapes
:
??:!"

_output_shapes	
:?:&#"
 
_output_shapes
:
??:!$

_output_shapes	
:?:&%"
 
_output_shapes
:
??:!&

_output_shapes	
:?:%'!

_output_shapes
:	?: (

_output_shapes
::)

_output_shapes
: 
̩
?
#__inference__traced_restore_2763782
file_prefix
assignvariableop_mean
assignvariableop_1_variance
assignvariableop_2_count'
#assignvariableop_3_dense_105_kernel%
!assignvariableop_4_dense_105_bias'
#assignvariableop_5_dense_106_kernel%
!assignvariableop_6_dense_106_bias'
#assignvariableop_7_dense_107_kernel%
!assignvariableop_8_dense_107_bias'
#assignvariableop_9_dense_108_kernel&
"assignvariableop_10_dense_108_bias(
$assignvariableop_11_dense_109_kernel&
"assignvariableop_12_dense_109_bias#
assignvariableop_13_adamax_iter%
!assignvariableop_14_adamax_beta_1%
!assignvariableop_15_adamax_beta_2$
 assignvariableop_16_adamax_decay,
(assignvariableop_17_adamax_learning_rate
assignvariableop_18_total
assignvariableop_19_count_11
-assignvariableop_20_adamax_dense_105_kernel_m/
+assignvariableop_21_adamax_dense_105_bias_m1
-assignvariableop_22_adamax_dense_106_kernel_m/
+assignvariableop_23_adamax_dense_106_bias_m1
-assignvariableop_24_adamax_dense_107_kernel_m/
+assignvariableop_25_adamax_dense_107_bias_m1
-assignvariableop_26_adamax_dense_108_kernel_m/
+assignvariableop_27_adamax_dense_108_bias_m1
-assignvariableop_28_adamax_dense_109_kernel_m/
+assignvariableop_29_adamax_dense_109_bias_m1
-assignvariableop_30_adamax_dense_105_kernel_v/
+assignvariableop_31_adamax_dense_105_bias_v1
-assignvariableop_32_adamax_dense_106_kernel_v/
+assignvariableop_33_adamax_dense_106_bias_v1
-assignvariableop_34_adamax_dense_107_kernel_v/
+assignvariableop_35_adamax_dense_107_bias_v1
-assignvariableop_36_adamax_dense_108_kernel_v/
+assignvariableop_37_adamax_dense_108_bias_v1
-assignvariableop_38_adamax_dense_109_kernel_v/
+assignvariableop_39_adamax_dense_109_bias_v
identity_41??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:)*
dtype0*?
value?B?)B4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:)*
dtype0*e
value\BZ)B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::*7
dtypes-
+2)		2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_meanIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_varianceIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_countIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp#assignvariableop_3_dense_105_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_105_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp#assignvariableop_5_dense_106_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_106_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp#assignvariableop_7_dense_107_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_107_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp#assignvariableop_9_dense_108_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_108_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp$assignvariableop_11_dense_109_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_109_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_adamax_iterIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp!assignvariableop_14_adamax_beta_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp!assignvariableop_15_adamax_beta_2Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp assignvariableop_16_adamax_decayIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp(assignvariableop_17_adamax_learning_rateIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpassignvariableop_18_totalIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpassignvariableop_19_count_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp-assignvariableop_20_adamax_dense_105_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp+assignvariableop_21_adamax_dense_105_bias_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp-assignvariableop_22_adamax_dense_106_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp+assignvariableop_23_adamax_dense_106_bias_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp-assignvariableop_24_adamax_dense_107_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp+assignvariableop_25_adamax_dense_107_bias_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp-assignvariableop_26_adamax_dense_108_kernel_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adamax_dense_108_bias_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp-assignvariableop_28_adamax_dense_109_kernel_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adamax_dense_109_bias_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp-assignvariableop_30_adamax_dense_105_kernel_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adamax_dense_105_bias_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp-assignvariableop_32_adamax_dense_106_kernel_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adamax_dense_106_bias_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp-assignvariableop_34_adamax_dense_107_kernel_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp+assignvariableop_35_adamax_dense_107_bias_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp-assignvariableop_36_adamax_dense_108_kernel_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp+assignvariableop_37_adamax_dense_108_bias_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp-assignvariableop_38_adamax_dense_109_kernel_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp+assignvariableop_39_adamax_dense_109_bias_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_399
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_40Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_40?
Identity_41IdentityIdentity_40:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_41"#
identity_41Identity_41:output:0*?
_input_shapes?
?: ::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?	
?
F__inference_dense_106_layer_call_and_return_conditional_losses_2762929

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
+__inference_dense_109_layer_call_fn_2763509

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_109_layer_call_and_return_conditional_losses_27630092
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
/__inference_sequential_21_layer_call_fn_2763381

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_21_layer_call_and_return_conditional_losses_27631132
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:??????????????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?
?
+__inference_dense_105_layer_call_fn_2763430

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_105_layer_call_and_return_conditional_losses_27629022
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?-
?
J__inference_sequential_21_layer_call_and_return_conditional_losses_2763068
normalization_input1
-normalization_reshape_readvariableop_resource3
/normalization_reshape_1_readvariableop_resource
dense_105_2763042
dense_105_2763044
dense_106_2763047
dense_106_2763049
dense_107_2763052
dense_107_2763054
dense_108_2763057
dense_108_2763059
dense_109_2763062
dense_109_2763064
identity??!dense_105/StatefulPartitionedCall?!dense_106/StatefulPartitionedCall?!dense_107/StatefulPartitionedCall?!dense_108/StatefulPartitionedCall?!dense_109/StatefulPartitionedCall?$normalization/Reshape/ReadVariableOp?&normalization/Reshape_1/ReadVariableOp?
$normalization/Reshape/ReadVariableOpReadVariableOp-normalization_reshape_readvariableop_resource*
_output_shapes
:*
dtype02&
$normalization/Reshape/ReadVariableOp?
normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization/Reshape/shape?
normalization/ReshapeReshape,normalization/Reshape/ReadVariableOp:value:0$normalization/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization/Reshape?
&normalization/Reshape_1/ReadVariableOpReadVariableOp/normalization_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02(
&normalization/Reshape_1/ReadVariableOp?
normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization/Reshape_1/shape?
normalization/Reshape_1Reshape.normalization/Reshape_1/ReadVariableOp:value:0&normalization/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization/Reshape_1?
normalization/subSubnormalization_inputnormalization/Reshape:output:0*
T0*'
_output_shapes
:?????????2
normalization/sub{
normalization/SqrtSqrt normalization/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization/Sqrtw
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization/Maximum/y?
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization/Maximum?
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization/truediv?
!dense_105/StatefulPartitionedCallStatefulPartitionedCallnormalization/truediv:z:0dense_105_2763042dense_105_2763044*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_105_layer_call_and_return_conditional_losses_27629022#
!dense_105/StatefulPartitionedCall?
!dense_106/StatefulPartitionedCallStatefulPartitionedCall*dense_105/StatefulPartitionedCall:output:0dense_106_2763047dense_106_2763049*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_106_layer_call_and_return_conditional_losses_27629292#
!dense_106/StatefulPartitionedCall?
!dense_107/StatefulPartitionedCallStatefulPartitionedCall*dense_106/StatefulPartitionedCall:output:0dense_107_2763052dense_107_2763054*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_107_layer_call_and_return_conditional_losses_27629562#
!dense_107/StatefulPartitionedCall?
!dense_108/StatefulPartitionedCallStatefulPartitionedCall*dense_107/StatefulPartitionedCall:output:0dense_108_2763057dense_108_2763059*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_108_layer_call_and_return_conditional_losses_27629832#
!dense_108/StatefulPartitionedCall?
!dense_109/StatefulPartitionedCallStatefulPartitionedCall*dense_108/StatefulPartitionedCall:output:0dense_109_2763062dense_109_2763064*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_109_layer_call_and_return_conditional_losses_27630092#
!dense_109/StatefulPartitionedCall?
IdentityIdentity*dense_109/StatefulPartitionedCall:output:0"^dense_105/StatefulPartitionedCall"^dense_106/StatefulPartitionedCall"^dense_107/StatefulPartitionedCall"^dense_108/StatefulPartitionedCall"^dense_109/StatefulPartitionedCall%^normalization/Reshape/ReadVariableOp'^normalization/Reshape_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:??????????????????::::::::::::2F
!dense_105/StatefulPartitionedCall!dense_105/StatefulPartitionedCall2F
!dense_106/StatefulPartitionedCall!dense_106/StatefulPartitionedCall2F
!dense_107/StatefulPartitionedCall!dense_107/StatefulPartitionedCall2F
!dense_108/StatefulPartitionedCall!dense_108/StatefulPartitionedCall2F
!dense_109/StatefulPartitionedCall!dense_109/StatefulPartitionedCall2L
$normalization/Reshape/ReadVariableOp$normalization/Reshape/ReadVariableOp2P
&normalization/Reshape_1/ReadVariableOp&normalization/Reshape_1/ReadVariableOp:e a
0
_output_shapes
:??????????????????
-
_user_specified_namenormalization_input
?	
?
F__inference_dense_105_layer_call_and_return_conditional_losses_2763421

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?-
?
J__inference_sequential_21_layer_call_and_return_conditional_losses_2763026
normalization_input1
-normalization_reshape_readvariableop_resource3
/normalization_reshape_1_readvariableop_resource
dense_105_2762913
dense_105_2762915
dense_106_2762940
dense_106_2762942
dense_107_2762967
dense_107_2762969
dense_108_2762994
dense_108_2762996
dense_109_2763020
dense_109_2763022
identity??!dense_105/StatefulPartitionedCall?!dense_106/StatefulPartitionedCall?!dense_107/StatefulPartitionedCall?!dense_108/StatefulPartitionedCall?!dense_109/StatefulPartitionedCall?$normalization/Reshape/ReadVariableOp?&normalization/Reshape_1/ReadVariableOp?
$normalization/Reshape/ReadVariableOpReadVariableOp-normalization_reshape_readvariableop_resource*
_output_shapes
:*
dtype02&
$normalization/Reshape/ReadVariableOp?
normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization/Reshape/shape?
normalization/ReshapeReshape,normalization/Reshape/ReadVariableOp:value:0$normalization/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization/Reshape?
&normalization/Reshape_1/ReadVariableOpReadVariableOp/normalization_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02(
&normalization/Reshape_1/ReadVariableOp?
normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization/Reshape_1/shape?
normalization/Reshape_1Reshape.normalization/Reshape_1/ReadVariableOp:value:0&normalization/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization/Reshape_1?
normalization/subSubnormalization_inputnormalization/Reshape:output:0*
T0*'
_output_shapes
:?????????2
normalization/sub{
normalization/SqrtSqrt normalization/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization/Sqrtw
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization/Maximum/y?
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization/Maximum?
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization/truediv?
!dense_105/StatefulPartitionedCallStatefulPartitionedCallnormalization/truediv:z:0dense_105_2762913dense_105_2762915*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_105_layer_call_and_return_conditional_losses_27629022#
!dense_105/StatefulPartitionedCall?
!dense_106/StatefulPartitionedCallStatefulPartitionedCall*dense_105/StatefulPartitionedCall:output:0dense_106_2762940dense_106_2762942*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_106_layer_call_and_return_conditional_losses_27629292#
!dense_106/StatefulPartitionedCall?
!dense_107/StatefulPartitionedCallStatefulPartitionedCall*dense_106/StatefulPartitionedCall:output:0dense_107_2762967dense_107_2762969*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_107_layer_call_and_return_conditional_losses_27629562#
!dense_107/StatefulPartitionedCall?
!dense_108/StatefulPartitionedCallStatefulPartitionedCall*dense_107/StatefulPartitionedCall:output:0dense_108_2762994dense_108_2762996*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_108_layer_call_and_return_conditional_losses_27629832#
!dense_108/StatefulPartitionedCall?
!dense_109/StatefulPartitionedCallStatefulPartitionedCall*dense_108/StatefulPartitionedCall:output:0dense_109_2763020dense_109_2763022*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_109_layer_call_and_return_conditional_losses_27630092#
!dense_109/StatefulPartitionedCall?
IdentityIdentity*dense_109/StatefulPartitionedCall:output:0"^dense_105/StatefulPartitionedCall"^dense_106/StatefulPartitionedCall"^dense_107/StatefulPartitionedCall"^dense_108/StatefulPartitionedCall"^dense_109/StatefulPartitionedCall%^normalization/Reshape/ReadVariableOp'^normalization/Reshape_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:??????????????????::::::::::::2F
!dense_105/StatefulPartitionedCall!dense_105/StatefulPartitionedCall2F
!dense_106/StatefulPartitionedCall!dense_106/StatefulPartitionedCall2F
!dense_107/StatefulPartitionedCall!dense_107/StatefulPartitionedCall2F
!dense_108/StatefulPartitionedCall!dense_108/StatefulPartitionedCall2F
!dense_109/StatefulPartitionedCall!dense_109/StatefulPartitionedCall2L
$normalization/Reshape/ReadVariableOp$normalization/Reshape/ReadVariableOp2P
&normalization/Reshape_1/ReadVariableOp&normalization/Reshape_1/ReadVariableOp:e a
0
_output_shapes
:??????????????????
-
_user_specified_namenormalization_input
?	
?
/__inference_sequential_21_layer_call_fn_2763410

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_21_layer_call_and_return_conditional_losses_27631842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:??????????????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?	
?
/__inference_sequential_21_layer_call_fn_2763140
normalization_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallnormalization_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_21_layer_call_and_return_conditional_losses_27631132
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:??????????????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:e a
0
_output_shapes
:??????????????????
-
_user_specified_namenormalization_input
?-
?
J__inference_sequential_21_layer_call_and_return_conditional_losses_2763113

inputs1
-normalization_reshape_readvariableop_resource3
/normalization_reshape_1_readvariableop_resource
dense_105_2763087
dense_105_2763089
dense_106_2763092
dense_106_2763094
dense_107_2763097
dense_107_2763099
dense_108_2763102
dense_108_2763104
dense_109_2763107
dense_109_2763109
identity??!dense_105/StatefulPartitionedCall?!dense_106/StatefulPartitionedCall?!dense_107/StatefulPartitionedCall?!dense_108/StatefulPartitionedCall?!dense_109/StatefulPartitionedCall?$normalization/Reshape/ReadVariableOp?&normalization/Reshape_1/ReadVariableOp?
$normalization/Reshape/ReadVariableOpReadVariableOp-normalization_reshape_readvariableop_resource*
_output_shapes
:*
dtype02&
$normalization/Reshape/ReadVariableOp?
normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization/Reshape/shape?
normalization/ReshapeReshape,normalization/Reshape/ReadVariableOp:value:0$normalization/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization/Reshape?
&normalization/Reshape_1/ReadVariableOpReadVariableOp/normalization_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02(
&normalization/Reshape_1/ReadVariableOp?
normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization/Reshape_1/shape?
normalization/Reshape_1Reshape.normalization/Reshape_1/ReadVariableOp:value:0&normalization/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization/Reshape_1?
normalization/subSubinputsnormalization/Reshape:output:0*
T0*'
_output_shapes
:?????????2
normalization/sub{
normalization/SqrtSqrt normalization/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization/Sqrtw
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization/Maximum/y?
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization/Maximum?
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization/truediv?
!dense_105/StatefulPartitionedCallStatefulPartitionedCallnormalization/truediv:z:0dense_105_2763087dense_105_2763089*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_105_layer_call_and_return_conditional_losses_27629022#
!dense_105/StatefulPartitionedCall?
!dense_106/StatefulPartitionedCallStatefulPartitionedCall*dense_105/StatefulPartitionedCall:output:0dense_106_2763092dense_106_2763094*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_106_layer_call_and_return_conditional_losses_27629292#
!dense_106/StatefulPartitionedCall?
!dense_107/StatefulPartitionedCallStatefulPartitionedCall*dense_106/StatefulPartitionedCall:output:0dense_107_2763097dense_107_2763099*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_107_layer_call_and_return_conditional_losses_27629562#
!dense_107/StatefulPartitionedCall?
!dense_108/StatefulPartitionedCallStatefulPartitionedCall*dense_107/StatefulPartitionedCall:output:0dense_108_2763102dense_108_2763104*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_108_layer_call_and_return_conditional_losses_27629832#
!dense_108/StatefulPartitionedCall?
!dense_109/StatefulPartitionedCallStatefulPartitionedCall*dense_108/StatefulPartitionedCall:output:0dense_109_2763107dense_109_2763109*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_109_layer_call_and_return_conditional_losses_27630092#
!dense_109/StatefulPartitionedCall?
IdentityIdentity*dense_109/StatefulPartitionedCall:output:0"^dense_105/StatefulPartitionedCall"^dense_106/StatefulPartitionedCall"^dense_107/StatefulPartitionedCall"^dense_108/StatefulPartitionedCall"^dense_109/StatefulPartitionedCall%^normalization/Reshape/ReadVariableOp'^normalization/Reshape_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:??????????????????::::::::::::2F
!dense_105/StatefulPartitionedCall!dense_105/StatefulPartitionedCall2F
!dense_106/StatefulPartitionedCall!dense_106/StatefulPartitionedCall2F
!dense_107/StatefulPartitionedCall!dense_107/StatefulPartitionedCall2F
!dense_108/StatefulPartitionedCall!dense_108/StatefulPartitionedCall2F
!dense_109/StatefulPartitionedCall!dense_109/StatefulPartitionedCall2L
$normalization/Reshape/ReadVariableOp$normalization/Reshape/ReadVariableOp2P
&normalization/Reshape_1/ReadVariableOp&normalization/Reshape_1/ReadVariableOp:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?	
?
F__inference_dense_105_layer_call_and_return_conditional_losses_2762902

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?U
?
"__inference__wrapped_model_2762874
normalization_input?
;sequential_21_normalization_reshape_readvariableop_resourceA
=sequential_21_normalization_reshape_1_readvariableop_resource:
6sequential_21_dense_105_matmul_readvariableop_resource;
7sequential_21_dense_105_biasadd_readvariableop_resource:
6sequential_21_dense_106_matmul_readvariableop_resource;
7sequential_21_dense_106_biasadd_readvariableop_resource:
6sequential_21_dense_107_matmul_readvariableop_resource;
7sequential_21_dense_107_biasadd_readvariableop_resource:
6sequential_21_dense_108_matmul_readvariableop_resource;
7sequential_21_dense_108_biasadd_readvariableop_resource:
6sequential_21_dense_109_matmul_readvariableop_resource;
7sequential_21_dense_109_biasadd_readvariableop_resource
identity??.sequential_21/dense_105/BiasAdd/ReadVariableOp?-sequential_21/dense_105/MatMul/ReadVariableOp?.sequential_21/dense_106/BiasAdd/ReadVariableOp?-sequential_21/dense_106/MatMul/ReadVariableOp?.sequential_21/dense_107/BiasAdd/ReadVariableOp?-sequential_21/dense_107/MatMul/ReadVariableOp?.sequential_21/dense_108/BiasAdd/ReadVariableOp?-sequential_21/dense_108/MatMul/ReadVariableOp?.sequential_21/dense_109/BiasAdd/ReadVariableOp?-sequential_21/dense_109/MatMul/ReadVariableOp?2sequential_21/normalization/Reshape/ReadVariableOp?4sequential_21/normalization/Reshape_1/ReadVariableOp?
2sequential_21/normalization/Reshape/ReadVariableOpReadVariableOp;sequential_21_normalization_reshape_readvariableop_resource*
_output_shapes
:*
dtype024
2sequential_21/normalization/Reshape/ReadVariableOp?
)sequential_21/normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2+
)sequential_21/normalization/Reshape/shape?
#sequential_21/normalization/ReshapeReshape:sequential_21/normalization/Reshape/ReadVariableOp:value:02sequential_21/normalization/Reshape/shape:output:0*
T0*
_output_shapes

:2%
#sequential_21/normalization/Reshape?
4sequential_21/normalization/Reshape_1/ReadVariableOpReadVariableOp=sequential_21_normalization_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype026
4sequential_21/normalization/Reshape_1/ReadVariableOp?
+sequential_21/normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2-
+sequential_21/normalization/Reshape_1/shape?
%sequential_21/normalization/Reshape_1Reshape<sequential_21/normalization/Reshape_1/ReadVariableOp:value:04sequential_21/normalization/Reshape_1/shape:output:0*
T0*
_output_shapes

:2'
%sequential_21/normalization/Reshape_1?
sequential_21/normalization/subSubnormalization_input,sequential_21/normalization/Reshape:output:0*
T0*'
_output_shapes
:?????????2!
sequential_21/normalization/sub?
 sequential_21/normalization/SqrtSqrt.sequential_21/normalization/Reshape_1:output:0*
T0*
_output_shapes

:2"
 sequential_21/normalization/Sqrt?
%sequential_21/normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32'
%sequential_21/normalization/Maximum/y?
#sequential_21/normalization/MaximumMaximum$sequential_21/normalization/Sqrt:y:0.sequential_21/normalization/Maximum/y:output:0*
T0*
_output_shapes

:2%
#sequential_21/normalization/Maximum?
#sequential_21/normalization/truedivRealDiv#sequential_21/normalization/sub:z:0'sequential_21/normalization/Maximum:z:0*
T0*'
_output_shapes
:?????????2%
#sequential_21/normalization/truediv?
-sequential_21/dense_105/MatMul/ReadVariableOpReadVariableOp6sequential_21_dense_105_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02/
-sequential_21/dense_105/MatMul/ReadVariableOp?
sequential_21/dense_105/MatMulMatMul'sequential_21/normalization/truediv:z:05sequential_21/dense_105/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
sequential_21/dense_105/MatMul?
.sequential_21/dense_105/BiasAdd/ReadVariableOpReadVariableOp7sequential_21_dense_105_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.sequential_21/dense_105/BiasAdd/ReadVariableOp?
sequential_21/dense_105/BiasAddBiasAdd(sequential_21/dense_105/MatMul:product:06sequential_21/dense_105/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
sequential_21/dense_105/BiasAdd?
sequential_21/dense_105/ReluRelu(sequential_21/dense_105/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential_21/dense_105/Relu?
-sequential_21/dense_106/MatMul/ReadVariableOpReadVariableOp6sequential_21_dense_106_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02/
-sequential_21/dense_106/MatMul/ReadVariableOp?
sequential_21/dense_106/MatMulMatMul*sequential_21/dense_105/Relu:activations:05sequential_21/dense_106/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
sequential_21/dense_106/MatMul?
.sequential_21/dense_106/BiasAdd/ReadVariableOpReadVariableOp7sequential_21_dense_106_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.sequential_21/dense_106/BiasAdd/ReadVariableOp?
sequential_21/dense_106/BiasAddBiasAdd(sequential_21/dense_106/MatMul:product:06sequential_21/dense_106/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
sequential_21/dense_106/BiasAdd?
sequential_21/dense_106/ReluRelu(sequential_21/dense_106/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential_21/dense_106/Relu?
-sequential_21/dense_107/MatMul/ReadVariableOpReadVariableOp6sequential_21_dense_107_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02/
-sequential_21/dense_107/MatMul/ReadVariableOp?
sequential_21/dense_107/MatMulMatMul*sequential_21/dense_106/Relu:activations:05sequential_21/dense_107/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
sequential_21/dense_107/MatMul?
.sequential_21/dense_107/BiasAdd/ReadVariableOpReadVariableOp7sequential_21_dense_107_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.sequential_21/dense_107/BiasAdd/ReadVariableOp?
sequential_21/dense_107/BiasAddBiasAdd(sequential_21/dense_107/MatMul:product:06sequential_21/dense_107/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
sequential_21/dense_107/BiasAdd?
sequential_21/dense_107/ReluRelu(sequential_21/dense_107/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential_21/dense_107/Relu?
-sequential_21/dense_108/MatMul/ReadVariableOpReadVariableOp6sequential_21_dense_108_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02/
-sequential_21/dense_108/MatMul/ReadVariableOp?
sequential_21/dense_108/MatMulMatMul*sequential_21/dense_107/Relu:activations:05sequential_21/dense_108/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
sequential_21/dense_108/MatMul?
.sequential_21/dense_108/BiasAdd/ReadVariableOpReadVariableOp7sequential_21_dense_108_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.sequential_21/dense_108/BiasAdd/ReadVariableOp?
sequential_21/dense_108/BiasAddBiasAdd(sequential_21/dense_108/MatMul:product:06sequential_21/dense_108/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
sequential_21/dense_108/BiasAdd?
sequential_21/dense_108/ReluRelu(sequential_21/dense_108/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential_21/dense_108/Relu?
-sequential_21/dense_109/MatMul/ReadVariableOpReadVariableOp6sequential_21_dense_109_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02/
-sequential_21/dense_109/MatMul/ReadVariableOp?
sequential_21/dense_109/MatMulMatMul*sequential_21/dense_108/Relu:activations:05sequential_21/dense_109/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2 
sequential_21/dense_109/MatMul?
.sequential_21/dense_109/BiasAdd/ReadVariableOpReadVariableOp7sequential_21_dense_109_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.sequential_21/dense_109/BiasAdd/ReadVariableOp?
sequential_21/dense_109/BiasAddBiasAdd(sequential_21/dense_109/MatMul:product:06sequential_21/dense_109/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2!
sequential_21/dense_109/BiasAdd?
IdentityIdentity(sequential_21/dense_109/BiasAdd:output:0/^sequential_21/dense_105/BiasAdd/ReadVariableOp.^sequential_21/dense_105/MatMul/ReadVariableOp/^sequential_21/dense_106/BiasAdd/ReadVariableOp.^sequential_21/dense_106/MatMul/ReadVariableOp/^sequential_21/dense_107/BiasAdd/ReadVariableOp.^sequential_21/dense_107/MatMul/ReadVariableOp/^sequential_21/dense_108/BiasAdd/ReadVariableOp.^sequential_21/dense_108/MatMul/ReadVariableOp/^sequential_21/dense_109/BiasAdd/ReadVariableOp.^sequential_21/dense_109/MatMul/ReadVariableOp3^sequential_21/normalization/Reshape/ReadVariableOp5^sequential_21/normalization/Reshape_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:??????????????????::::::::::::2`
.sequential_21/dense_105/BiasAdd/ReadVariableOp.sequential_21/dense_105/BiasAdd/ReadVariableOp2^
-sequential_21/dense_105/MatMul/ReadVariableOp-sequential_21/dense_105/MatMul/ReadVariableOp2`
.sequential_21/dense_106/BiasAdd/ReadVariableOp.sequential_21/dense_106/BiasAdd/ReadVariableOp2^
-sequential_21/dense_106/MatMul/ReadVariableOp-sequential_21/dense_106/MatMul/ReadVariableOp2`
.sequential_21/dense_107/BiasAdd/ReadVariableOp.sequential_21/dense_107/BiasAdd/ReadVariableOp2^
-sequential_21/dense_107/MatMul/ReadVariableOp-sequential_21/dense_107/MatMul/ReadVariableOp2`
.sequential_21/dense_108/BiasAdd/ReadVariableOp.sequential_21/dense_108/BiasAdd/ReadVariableOp2^
-sequential_21/dense_108/MatMul/ReadVariableOp-sequential_21/dense_108/MatMul/ReadVariableOp2`
.sequential_21/dense_109/BiasAdd/ReadVariableOp.sequential_21/dense_109/BiasAdd/ReadVariableOp2^
-sequential_21/dense_109/MatMul/ReadVariableOp-sequential_21/dense_109/MatMul/ReadVariableOp2h
2sequential_21/normalization/Reshape/ReadVariableOp2sequential_21/normalization/Reshape/ReadVariableOp2l
4sequential_21/normalization/Reshape_1/ReadVariableOp4sequential_21/normalization/Reshape_1/ReadVariableOp:e a
0
_output_shapes
:??????????????????
-
_user_specified_namenormalization_input
?	
?
F__inference_dense_106_layer_call_and_return_conditional_losses_2763441

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
F__inference_dense_108_layer_call_and_return_conditional_losses_2762983

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
F__inference_dense_108_layer_call_and_return_conditional_losses_2763481

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
/__inference_sequential_21_layer_call_fn_2763211
normalization_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallnormalization_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_21_layer_call_and_return_conditional_losses_27631842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:??????????????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:e a
0
_output_shapes
:??????????????????
-
_user_specified_namenormalization_input
?A
?
J__inference_sequential_21_layer_call_and_return_conditional_losses_2763352

inputs1
-normalization_reshape_readvariableop_resource3
/normalization_reshape_1_readvariableop_resource,
(dense_105_matmul_readvariableop_resource-
)dense_105_biasadd_readvariableop_resource,
(dense_106_matmul_readvariableop_resource-
)dense_106_biasadd_readvariableop_resource,
(dense_107_matmul_readvariableop_resource-
)dense_107_biasadd_readvariableop_resource,
(dense_108_matmul_readvariableop_resource-
)dense_108_biasadd_readvariableop_resource,
(dense_109_matmul_readvariableop_resource-
)dense_109_biasadd_readvariableop_resource
identity?? dense_105/BiasAdd/ReadVariableOp?dense_105/MatMul/ReadVariableOp? dense_106/BiasAdd/ReadVariableOp?dense_106/MatMul/ReadVariableOp? dense_107/BiasAdd/ReadVariableOp?dense_107/MatMul/ReadVariableOp? dense_108/BiasAdd/ReadVariableOp?dense_108/MatMul/ReadVariableOp? dense_109/BiasAdd/ReadVariableOp?dense_109/MatMul/ReadVariableOp?$normalization/Reshape/ReadVariableOp?&normalization/Reshape_1/ReadVariableOp?
$normalization/Reshape/ReadVariableOpReadVariableOp-normalization_reshape_readvariableop_resource*
_output_shapes
:*
dtype02&
$normalization/Reshape/ReadVariableOp?
normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization/Reshape/shape?
normalization/ReshapeReshape,normalization/Reshape/ReadVariableOp:value:0$normalization/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization/Reshape?
&normalization/Reshape_1/ReadVariableOpReadVariableOp/normalization_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02(
&normalization/Reshape_1/ReadVariableOp?
normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization/Reshape_1/shape?
normalization/Reshape_1Reshape.normalization/Reshape_1/ReadVariableOp:value:0&normalization/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization/Reshape_1?
normalization/subSubinputsnormalization/Reshape:output:0*
T0*'
_output_shapes
:?????????2
normalization/sub{
normalization/SqrtSqrt normalization/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization/Sqrtw
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization/Maximum/y?
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization/Maximum?
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization/truediv?
dense_105/MatMul/ReadVariableOpReadVariableOp(dense_105_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02!
dense_105/MatMul/ReadVariableOp?
dense_105/MatMulMatMulnormalization/truediv:z:0'dense_105/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_105/MatMul?
 dense_105/BiasAdd/ReadVariableOpReadVariableOp)dense_105_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_105/BiasAdd/ReadVariableOp?
dense_105/BiasAddBiasAdddense_105/MatMul:product:0(dense_105/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_105/BiasAddw
dense_105/ReluReludense_105/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_105/Relu?
dense_106/MatMul/ReadVariableOpReadVariableOp(dense_106_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02!
dense_106/MatMul/ReadVariableOp?
dense_106/MatMulMatMuldense_105/Relu:activations:0'dense_106/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_106/MatMul?
 dense_106/BiasAdd/ReadVariableOpReadVariableOp)dense_106_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_106/BiasAdd/ReadVariableOp?
dense_106/BiasAddBiasAdddense_106/MatMul:product:0(dense_106/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_106/BiasAddw
dense_106/ReluReludense_106/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_106/Relu?
dense_107/MatMul/ReadVariableOpReadVariableOp(dense_107_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02!
dense_107/MatMul/ReadVariableOp?
dense_107/MatMulMatMuldense_106/Relu:activations:0'dense_107/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_107/MatMul?
 dense_107/BiasAdd/ReadVariableOpReadVariableOp)dense_107_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_107/BiasAdd/ReadVariableOp?
dense_107/BiasAddBiasAdddense_107/MatMul:product:0(dense_107/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_107/BiasAddw
dense_107/ReluReludense_107/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_107/Relu?
dense_108/MatMul/ReadVariableOpReadVariableOp(dense_108_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02!
dense_108/MatMul/ReadVariableOp?
dense_108/MatMulMatMuldense_107/Relu:activations:0'dense_108/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_108/MatMul?
 dense_108/BiasAdd/ReadVariableOpReadVariableOp)dense_108_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_108/BiasAdd/ReadVariableOp?
dense_108/BiasAddBiasAdddense_108/MatMul:product:0(dense_108/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_108/BiasAddw
dense_108/ReluReludense_108/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_108/Relu?
dense_109/MatMul/ReadVariableOpReadVariableOp(dense_109_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02!
dense_109/MatMul/ReadVariableOp?
dense_109/MatMulMatMuldense_108/Relu:activations:0'dense_109/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_109/MatMul?
 dense_109/BiasAdd/ReadVariableOpReadVariableOp)dense_109_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_109/BiasAdd/ReadVariableOp?
dense_109/BiasAddBiasAdddense_109/MatMul:product:0(dense_109/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_109/BiasAdd?
IdentityIdentitydense_109/BiasAdd:output:0!^dense_105/BiasAdd/ReadVariableOp ^dense_105/MatMul/ReadVariableOp!^dense_106/BiasAdd/ReadVariableOp ^dense_106/MatMul/ReadVariableOp!^dense_107/BiasAdd/ReadVariableOp ^dense_107/MatMul/ReadVariableOp!^dense_108/BiasAdd/ReadVariableOp ^dense_108/MatMul/ReadVariableOp!^dense_109/BiasAdd/ReadVariableOp ^dense_109/MatMul/ReadVariableOp%^normalization/Reshape/ReadVariableOp'^normalization/Reshape_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:??????????????????::::::::::::2D
 dense_105/BiasAdd/ReadVariableOp dense_105/BiasAdd/ReadVariableOp2B
dense_105/MatMul/ReadVariableOpdense_105/MatMul/ReadVariableOp2D
 dense_106/BiasAdd/ReadVariableOp dense_106/BiasAdd/ReadVariableOp2B
dense_106/MatMul/ReadVariableOpdense_106/MatMul/ReadVariableOp2D
 dense_107/BiasAdd/ReadVariableOp dense_107/BiasAdd/ReadVariableOp2B
dense_107/MatMul/ReadVariableOpdense_107/MatMul/ReadVariableOp2D
 dense_108/BiasAdd/ReadVariableOp dense_108/BiasAdd/ReadVariableOp2B
dense_108/MatMul/ReadVariableOpdense_108/MatMul/ReadVariableOp2D
 dense_109/BiasAdd/ReadVariableOp dense_109/BiasAdd/ReadVariableOp2B
dense_109/MatMul/ReadVariableOpdense_109/MatMul/ReadVariableOp2L
$normalization/Reshape/ReadVariableOp$normalization/Reshape/ReadVariableOp2P
&normalization/Reshape_1/ReadVariableOp&normalization/Reshape_1/ReadVariableOp:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?
?
+__inference_dense_106_layer_call_fn_2763450

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_106_layer_call_and_return_conditional_losses_27629292
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
F__inference_dense_107_layer_call_and_return_conditional_losses_2762956

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
%__inference_signature_wrapper_2763250
normalization_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallnormalization_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__wrapped_model_27628742
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:??????????????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:e a
0
_output_shapes
:??????????????????
-
_user_specified_namenormalization_input"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
\
normalization_inputE
%serving_default_normalization_input:0??????????????????=
	dense_1090
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?3
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
layer_with_weights-5
layer-5
	optimizer
regularization_losses
	trainable_variables

	variables
	keras_api

signatures
*m&call_and_return_all_conditional_losses
n__call__
o_default_save_signature"?/
_tf_keras_sequential?/{"class_name": "Sequential", "name": "sequential_21", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_21", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "normalization_input"}}, {"class_name": "Normalization", "config": {"name": "normalization", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [-1]}}}, {"class_name": "Dense", "config": {"name": "dense_105", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "units": 150, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_106", "trainable": true, "dtype": "float32", "units": 150, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_107", "trainable": true, "dtype": "float32", "units": 150, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_108", "trainable": true, "dtype": "float32", "units": 150, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_109", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_21", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "normalization_input"}}, {"class_name": "Normalization", "config": {"name": "normalization", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [-1]}}}, {"class_name": "Dense", "config": {"name": "dense_105", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "units": 150, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_106", "trainable": true, "dtype": "float32", "units": 150, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_107", "trainable": true, "dtype": "float32", "units": 150, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_108", "trainable": true, "dtype": "float32", "units": 150, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_109", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mean_squared_error", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adamax", "config": {"name": "Adamax", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07}}}}
?
state_variables
_broadcast_shape
mean
variance
	count
	keras_api"?
_tf_keras_layer?{"class_name": "Normalization", "name": "normalization", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": true, "config": {"name": "normalization", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [-1]}}, "build_input_shape": [512, 2]}
?

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
*p&call_and_return_all_conditional_losses
q__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_105", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_105", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "units": 150, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}}
?

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
*r&call_and_return_all_conditional_losses
s__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_106", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_106", "trainable": true, "dtype": "float32", "units": 150, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 150}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 150]}}
?

kernel
 bias
!regularization_losses
"trainable_variables
#	variables
$	keras_api
*t&call_and_return_all_conditional_losses
u__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_107", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_107", "trainable": true, "dtype": "float32", "units": 150, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 150}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 150]}}
?

%kernel
&bias
'regularization_losses
(trainable_variables
)	variables
*	keras_api
*v&call_and_return_all_conditional_losses
w__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_108", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_108", "trainable": true, "dtype": "float32", "units": 150, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 150}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 150]}}
?

+kernel
,bias
-regularization_losses
.trainable_variables
/	variables
0	keras_api
*x&call_and_return_all_conditional_losses
y__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_109", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_109", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 150}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 150]}}
?
1iter

2beta_1

3beta_2
	4decay
5learning_ratemYmZm[m\m] m^%m_&m`+ma,mbvcvdvevfvg vh%vi&vj+vk,vl"
	optimizer
 "
trackable_list_wrapper
f
0
1
2
3
4
 5
%6
&7
+8
,9"
trackable_list_wrapper
~
0
1
2
3
4
5
6
7
 8
%9
&10
+11
,12"
trackable_list_wrapper
?
6layer_metrics
7non_trainable_variables
8metrics
9layer_regularization_losses
regularization_losses
	trainable_variables

	variables

:layers
n__call__
o_default_save_signature
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses"
_generic_user_object
,
zserving_default"
signature_map
C
mean
variance
	count"
trackable_dict_wrapper
 "
trackable_list_wrapper
:2mean
:2variance
:	 2count
"
_generic_user_object
#:!	?2dense_105/kernel
:?2dense_105/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
;layer_metrics
<non_trainable_variables
=metrics
>layer_regularization_losses
regularization_losses
trainable_variables
	variables

?layers
q__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
$:"
??2dense_106/kernel
:?2dense_106/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
@layer_metrics
Anon_trainable_variables
Bmetrics
Clayer_regularization_losses
regularization_losses
trainable_variables
	variables

Dlayers
s__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
$:"
??2dense_107/kernel
:?2dense_107/bias
 "
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
?
Elayer_metrics
Fnon_trainable_variables
Gmetrics
Hlayer_regularization_losses
!regularization_losses
"trainable_variables
#	variables

Ilayers
u__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
$:"
??2dense_108/kernel
:?2dense_108/bias
 "
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
?
Jlayer_metrics
Knon_trainable_variables
Lmetrics
Mlayer_regularization_losses
'regularization_losses
(trainable_variables
)	variables

Nlayers
w__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
#:!	?2dense_109/kernel
:2dense_109/bias
 "
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
?
Olayer_metrics
Pnon_trainable_variables
Qmetrics
Rlayer_regularization_losses
-regularization_losses
.trainable_variables
/	variables

Slayers
y__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses"
_generic_user_object
:	 (2Adamax/iter
: (2Adamax/beta_1
: (2Adamax/beta_2
: (2Adamax/decay
: (2Adamax/learning_rate
 "
trackable_dict_wrapper
5
0
1
2"
trackable_list_wrapper
'
T0"
trackable_list_wrapper
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	Utotal
	Vcount
W	variables
X	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
:  (2total
:  (2count
.
U0
V1"
trackable_list_wrapper
-
W	variables"
_generic_user_object
*:(	?2Adamax/dense_105/kernel/m
$:"?2Adamax/dense_105/bias/m
+:)
??2Adamax/dense_106/kernel/m
$:"?2Adamax/dense_106/bias/m
+:)
??2Adamax/dense_107/kernel/m
$:"?2Adamax/dense_107/bias/m
+:)
??2Adamax/dense_108/kernel/m
$:"?2Adamax/dense_108/bias/m
*:(	?2Adamax/dense_109/kernel/m
#:!2Adamax/dense_109/bias/m
*:(	?2Adamax/dense_105/kernel/v
$:"?2Adamax/dense_105/bias/v
+:)
??2Adamax/dense_106/kernel/v
$:"?2Adamax/dense_106/bias/v
+:)
??2Adamax/dense_107/kernel/v
$:"?2Adamax/dense_107/bias/v
+:)
??2Adamax/dense_108/kernel/v
$:"?2Adamax/dense_108/bias/v
*:(	?2Adamax/dense_109/kernel/v
#:!2Adamax/dense_109/bias/v
?2?
J__inference_sequential_21_layer_call_and_return_conditional_losses_2763301
J__inference_sequential_21_layer_call_and_return_conditional_losses_2763068
J__inference_sequential_21_layer_call_and_return_conditional_losses_2763352
J__inference_sequential_21_layer_call_and_return_conditional_losses_2763026?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
/__inference_sequential_21_layer_call_fn_2763410
/__inference_sequential_21_layer_call_fn_2763381
/__inference_sequential_21_layer_call_fn_2763140
/__inference_sequential_21_layer_call_fn_2763211?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
"__inference__wrapped_model_2762874?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *;?8
6?3
normalization_input??????????????????
?2?
F__inference_dense_105_layer_call_and_return_conditional_losses_2763421?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dense_105_layer_call_fn_2763430?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dense_106_layer_call_and_return_conditional_losses_2763441?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dense_106_layer_call_fn_2763450?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dense_107_layer_call_and_return_conditional_losses_2763461?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dense_107_layer_call_fn_2763470?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dense_108_layer_call_and_return_conditional_losses_2763481?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dense_108_layer_call_fn_2763490?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dense_109_layer_call_and_return_conditional_losses_2763500?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dense_109_layer_call_fn_2763509?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
%__inference_signature_wrapper_2763250normalization_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
"__inference__wrapped_model_2762874? %&+,E?B
;?8
6?3
normalization_input??????????????????
? "5?2
0
	dense_109#? 
	dense_109??????????
F__inference_dense_105_layer_call_and_return_conditional_losses_2763421]/?,
%?"
 ?
inputs?????????
? "&?#
?
0??????????
? 
+__inference_dense_105_layer_call_fn_2763430P/?,
%?"
 ?
inputs?????????
? "????????????
F__inference_dense_106_layer_call_and_return_conditional_losses_2763441^0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
+__inference_dense_106_layer_call_fn_2763450Q0?-
&?#
!?
inputs??????????
? "????????????
F__inference_dense_107_layer_call_and_return_conditional_losses_2763461^ 0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
+__inference_dense_107_layer_call_fn_2763470Q 0?-
&?#
!?
inputs??????????
? "????????????
F__inference_dense_108_layer_call_and_return_conditional_losses_2763481^%&0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
+__inference_dense_108_layer_call_fn_2763490Q%&0?-
&?#
!?
inputs??????????
? "????????????
F__inference_dense_109_layer_call_and_return_conditional_losses_2763500]+,0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? 
+__inference_dense_109_layer_call_fn_2763509P+,0?-
&?#
!?
inputs??????????
? "???????????
J__inference_sequential_21_layer_call_and_return_conditional_losses_2763026? %&+,M?J
C?@
6?3
normalization_input??????????????????
p

 
? "%?"
?
0?????????
? ?
J__inference_sequential_21_layer_call_and_return_conditional_losses_2763068? %&+,M?J
C?@
6?3
normalization_input??????????????????
p 

 
? "%?"
?
0?????????
? ?
J__inference_sequential_21_layer_call_and_return_conditional_losses_2763301w %&+,@?=
6?3
)?&
inputs??????????????????
p

 
? "%?"
?
0?????????
? ?
J__inference_sequential_21_layer_call_and_return_conditional_losses_2763352w %&+,@?=
6?3
)?&
inputs??????????????????
p 

 
? "%?"
?
0?????????
? ?
/__inference_sequential_21_layer_call_fn_2763140w %&+,M?J
C?@
6?3
normalization_input??????????????????
p

 
? "???????????
/__inference_sequential_21_layer_call_fn_2763211w %&+,M?J
C?@
6?3
normalization_input??????????????????
p 

 
? "???????????
/__inference_sequential_21_layer_call_fn_2763381j %&+,@?=
6?3
)?&
inputs??????????????????
p

 
? "???????????
/__inference_sequential_21_layer_call_fn_2763410j %&+,@?=
6?3
)?&
inputs??????????????????
p 

 
? "???????????
%__inference_signature_wrapper_2763250? %&+,\?Y
? 
R?O
M
normalization_input6?3
normalization_input??????????????????"5?2
0
	dense_109#? 
	dense_109?????????