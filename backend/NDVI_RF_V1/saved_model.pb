ÖŽ
ę
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource
.
Identity

input"T
output"T"	
Ttype

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
ł
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
f
SimpleMLCreateModelResource
model_handle"
	containerstring "
shared_namestring 
á
SimpleMLInferenceOpWithHandle
numerical_features
boolean_features
categorical_int_features'
#categorical_set_int_features_values1
-categorical_set_int_features_row_splits_dim_1	1
-categorical_set_int_features_row_splits_dim_2	
model_handle
dense_predictions
dense_col_representation"
dense_output_dimint(0
Ł
#SimpleMLLoadModelFromPathWithHandle
model_handle
path" 
output_typeslist(string)
 "
file_prefixstring " 
allow_slow_inferencebool(
Á
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
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
m
StaticRegexReplace	
input

output"
patternstring"
rewritestring"
replace_globalbool(
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
°
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 
9
VarIsInitializedOp
resource
is_initialized
"serve*2.18.02v2.18.0-rc2-4-g6550e4bd8028ř
W
asset_path_initializerPlaceholder*
_output_shapes
: *
dtype0*
shape: 

VariableVarHandleOp*
_class
loc:@Variable*
_output_shapes
: *

debug_name	Variable/*
dtype0*
shape: *
shared_name
Variable
a
)Variable/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable*
_output_shapes
: 
z
Variable/AssignAssignVariableOpVariableasset_path_initializer*&
 _has_manual_control_dependencies(*
dtype0
]
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
: *
dtype0
Y
asset_path_initializer_1Placeholder*
_output_shapes
: *
dtype0*
shape: 
¤

Variable_1VarHandleOp*
_class
loc:@Variable_1*
_output_shapes
: *

debug_nameVariable_1/*
dtype0*
shape: *
shared_name
Variable_1
e
+Variable_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_1*
_output_shapes
: 

Variable_1/AssignAssignVariableOp
Variable_1asset_path_initializer_1*&
 _has_manual_control_dependencies(*
dtype0
a
Variable_1/Read/ReadVariableOpReadVariableOp
Variable_1*
_output_shapes
: *
dtype0
Y
asset_path_initializer_2Placeholder*
_output_shapes
: *
dtype0*
shape: 
¤

Variable_2VarHandleOp*
_class
loc:@Variable_2*
_output_shapes
: *

debug_nameVariable_2/*
dtype0*
shape: *
shared_name
Variable_2
e
+Variable_2/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_2*
_output_shapes
: 

Variable_2/AssignAssignVariableOp
Variable_2asset_path_initializer_2*&
 _has_manual_control_dependencies(*
dtype0
a
Variable_2/Read/ReadVariableOpReadVariableOp
Variable_2*
_output_shapes
: *
dtype0
Y
asset_path_initializer_3Placeholder*
_output_shapes
: *
dtype0*
shape: 
¤

Variable_3VarHandleOp*
_class
loc:@Variable_3*
_output_shapes
: *

debug_nameVariable_3/*
dtype0*
shape: *
shared_name
Variable_3
e
+Variable_3/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_3*
_output_shapes
: 

Variable_3/AssignAssignVariableOp
Variable_3asset_path_initializer_3*&
 _has_manual_control_dependencies(*
dtype0
a
Variable_3/Read/ReadVariableOpReadVariableOp
Variable_3*
_output_shapes
: *
dtype0
Y
asset_path_initializer_4Placeholder*
_output_shapes
: *
dtype0*
shape: 
¤

Variable_4VarHandleOp*
_class
loc:@Variable_4*
_output_shapes
: *

debug_nameVariable_4/*
dtype0*
shape: *
shared_name
Variable_4
e
+Variable_4/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_4*
_output_shapes
: 

Variable_4/AssignAssignVariableOp
Variable_4asset_path_initializer_4*&
 _has_manual_control_dependencies(*
dtype0
a
Variable_4/Read/ReadVariableOpReadVariableOp
Variable_4*
_output_shapes
: *
dtype0
v
countVarHandleOp*
_output_shapes
: *

debug_namecount/*
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
v
totalVarHandleOp*
_output_shapes
: *

debug_nametotal/*
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0

SimpleMLCreateModelResourceSimpleMLCreateModelResource*
_output_shapes
: *E
shared_name64simple_ml_model_70553a36-9e93-4d8e-9413-5e8378ae7ed6

learning_rateVarHandleOp*
_output_shapes
: *

debug_namelearning_rate/*
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0

	iterationVarHandleOp*
_output_shapes
: *

debug_name
iteration/*
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	


is_trainedVarHandleOp*
_output_shapes
: *

debug_nameis_trained/*
dtype0
*
shape: *
shared_name
is_trained
a
is_trained/Read/ReadVariableOpReadVariableOp
is_trained*
_output_shapes
: *
dtype0

m
serving_default_B1Placeholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
m
serving_default_B2Placeholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
m
serving_default_B3Placeholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
m
serving_default_B4Placeholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
m
serving_default_B5Placeholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
m
serving_default_B6Placeholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
m
serving_default_B7Placeholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
n
serving_default_EVIPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
n
serving_default_NBRPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
o
serving_default_NDBIPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
p
serving_default_NDBaIPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
o
serving_default_NDMIPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
o
serving_default_NDVIPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
o
serving_default_NDWIPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
t
serving_default_elevationPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0	*
shape:˙˙˙˙˙˙˙˙˙

StatefulPartitionedCallStatefulPartitionedCallserving_default_B1serving_default_B2serving_default_B3serving_default_B4serving_default_B5serving_default_B6serving_default_B7serving_default_EVIserving_default_NBRserving_default_NDBIserving_default_NDBaIserving_default_NDMIserving_default_NDVIserving_default_NDWIserving_default_elevationSimpleMLCreateModelResource*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *5
config_proto%#

CPU

GPU2*0J 8 J *-
f(R&
$__inference_signature_wrapper_105649
a
ReadVariableOpReadVariableOpVariable^Variable/Assign*
_output_shapes
: *
dtype0
á
StatefulPartitionedCall_1StatefulPartitionedCallReadVariableOpSimpleMLCreateModelResource*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
 _has_manual_control_dependencies(*
_output_shapes
: * 
_read_only_resource_inputs
 *5
config_proto%#

CPU

GPU2*0J 8 J *(
f#R!
__inference__initializer_105660

NoOpNoOp^StatefulPartitionedCall_1^Variable/Assign^Variable_1/Assign^Variable_2/Assign^Variable_3/Assign^Variable_4/Assign

ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ç
value˝Bş Bł
Ą
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

_multitask
	_is_trained

_learner_params
	_features
	optimizer
loss
_models
_build_normalized_inputs
_finalize_predictions
call
call_get_leaves
yggdrasil_model_path_tensor

signatures*

	0*
* 
* 
°
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

trace_0
trace_1* 

trace_0
trace_1* 
* 
* 
JD
VARIABLE_VALUE
is_trained&_is_trained/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
O

_variables
_iterations
 _learning_rate
!_update_step_xla*
* 
	
"0* 

#trace_0
$trace_1* 

%trace_0* 

&trace_0
'trace_1* 
* 

(trace_0* 

)serving_default* 

	0*
* 

*0*
* 
* 
* 
* 
* 
* 

0*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
+
+_input_builder
,_compiled_model* 
* 
* 
* 
* 
* 

-	capture_0* 
* 
8
.	variables
/	keras_api
	0total
	1count*
P
2_feature_name_to_idx
3	_init_ops
#4categorical_str_to_int_hashmaps* 
S
5_model_loader
6_create_resource
7_initialize
8_destroy_resource* 
* 

00
11*

.	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
5
9_output_types
:
_all_files
-
_done_file* 

;trace_0* 

<trace_0* 

=trace_0* 
* 
%
>0
?1
-2
@3
A4* 
* 

-	capture_0* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ü
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename
is_trained	iterationlearning_ratetotalcountConst*
Tin
	2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *5
config_proto%#

CPU

GPU2*0J 8 J *(
f#R!
__inference__traced_save_105753
×
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filename
is_trained	iterationlearning_ratetotalcount*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *5
config_proto%#

CPU

GPU2*0J 8 J *+
f&R$
"__inference__traced_restore_105777â˛
ť
ô
*__inference__build_normalized_inputs_19890

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
	inputs_11
	inputs_12
inputs_9
	inputs_13
	inputs_10
	inputs_14	
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7

identity_8

identity_9
identity_10
identity_11
identity_12
identity_13
identity_14T
CastCast	inputs_14*

DstT0*

SrcT0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙J
IdentityIdentityinputs*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙N

Identity_1Identityinputs_1*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙N

Identity_2Identityinputs_2*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙N

Identity_3Identityinputs_3*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙N

Identity_4Identityinputs_4*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙N

Identity_5Identityinputs_5*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙N

Identity_6Identityinputs_6*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙N

Identity_7Identityinputs_7*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙N

Identity_8Identityinputs_8*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙O

Identity_9Identity	inputs_11*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙P
Identity_10Identity	inputs_12*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙O
Identity_11Identityinputs_9*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙P
Identity_12Identity	inputs_13*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙P
Identity_13Identity	inputs_10*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙O
Identity_14IdentityCast:y:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙"#
identity_10Identity_10:output:0"#
identity_11Identity_11:output:0"#
identity_12Identity_12:output:0"#
identity_13Identity_13:output:0"#
identity_14Identity_14:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"!

identity_9Identity_9:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ö
_input_shapesä
á:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K
G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K	G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ż

!__inference__wrapped_model_105497
b1
b2
b3
b4
b5
b6
b7
evi
nbr
ndbi	
ndbai
ndmi
ndvi
ndwi
	elevation	
random_forest_model_105493
identity˘+random_forest_model/StatefulPartitionedCall 
+random_forest_model/StatefulPartitionedCallStatefulPartitionedCallb1b2b3b4b5b6b7evinbrndbindbaindmindvindwi	elevationrandom_forest_model_105493*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *5
config_proto%#

CPU

GPU2*0J 8 J *
fR
__inference_call_19922
IdentityIdentity4random_forest_model/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
P
NoOpNoOp,^random_forest_model/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ř
_input_shapesć
ă:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: 2Z
+random_forest_model/StatefulPartitionedCall+random_forest_model/StatefulPartitionedCall:&"
 
_user_specified_name105493:NJ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	elevation:IE
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameNDWI:IE
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameNDVI:IE
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameNDMI:J
F
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameNDBaI:I	E
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameNDBI:HD
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameNBR:HD
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameEVI:GC
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameB7:GC
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameB6:GC
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameB5:GC
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameB4:GC
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameB3:GC
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameB2:G C
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameB1
Č
[
'__inference__finalize_predictions_19919
predictions
predictions_1
identityS
IdentityIdentitypredictions*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:˙˙˙˙˙˙˙˙˙
:
:GC

_output_shapes
:

%
_user_specified_namepredictions:T P
'
_output_shapes
:˙˙˙˙˙˙˙˙˙

%
_user_specified_namepredictions
ž
[
-__inference_yggdrasil_model_path_tensor_20218
staticregexreplace_input
identity
StaticRegexReplaceStaticRegexReplacestaticregexreplace_input*
_output_shapes
: *!
patterna7f2c213315f47eedone*
rewrite R
IdentityIdentityStaticRegexReplace:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
Ü
ú
4__inference_random_forest_model_layer_call_fn_105625
b1
b2
b3
b4
b5
b6
b7
evi
nbr
ndbi	
ndbai
ndmi
ndvi
ndwi
	elevation	
unknown
identity˘StatefulPartitionedCall˛
StatefulPartitionedCallStatefulPartitionedCallb1b2b3b4b5b6b7evinbrndbindbaindmindvindwi	elevationunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *5
config_proto%#

CPU

GPU2*0J 8 J *X
fSRQ
O__inference_random_forest_model_layer_call_and_return_conditional_losses_105583o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ř
_input_shapesć
ă:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name105621:NJ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	elevation:IE
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameNDWI:IE
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameNDVI:IE
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameNDMI:J
F
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameNDBaI:I	E
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameNDBI:HD
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameNBR:HD
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameEVI:GC
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameB7:GC
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameB6:GC
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameB5:GC
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameB4:GC
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameB3:GC
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameB2:G C
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameB1
ź

*__inference__build_normalized_inputs_20090
	inputs_b1
	inputs_b2
	inputs_b3
	inputs_b4
	inputs_b5
	inputs_b6
	inputs_b7

inputs_evi

inputs_nbr
inputs_ndbi
inputs_ndbai
inputs_ndmi
inputs_ndvi
inputs_ndwi
inputs_elevation	
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7

identity_8

identity_9
identity_10
identity_11
identity_12
identity_13
identity_14[
CastCastinputs_elevation*

DstT0*

SrcT0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙M
IdentityIdentity	inputs_b1*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙O

Identity_1Identity	inputs_b2*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙O

Identity_2Identity	inputs_b3*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙O

Identity_3Identity	inputs_b4*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙O

Identity_4Identity	inputs_b5*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙O

Identity_5Identity	inputs_b6*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙O

Identity_6Identity	inputs_b7*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙P

Identity_7Identity
inputs_evi*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙P

Identity_8Identity
inputs_nbr*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙Q

Identity_9Identityinputs_ndbi*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙S
Identity_10Identityinputs_ndbai*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙R
Identity_11Identityinputs_ndmi*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙R
Identity_12Identityinputs_ndvi*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙R
Identity_13Identityinputs_ndwi*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙O
Identity_14IdentityCast:y:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙"#
identity_10Identity_10:output:0"#
identity_11Identity_11:output:0"#
identity_12Identity_12:output:0"#
identity_13Identity_13:output:0"#
identity_14Identity_14:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"!

identity_9Identity_9:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ö
_input_shapesä
á:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:UQ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_user_specified_nameinputs_elevation:PL
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_nameinputs_ndwi:PL
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_nameinputs_ndvi:PL
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_nameinputs_ndmi:Q
M
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_nameinputs_ndbai:P	L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_nameinputs_ndbi:OK
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
inputs_nbr:OK
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
inputs_evi:NJ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	inputs_b7:NJ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	inputs_b6:NJ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	inputs_b5:NJ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	inputs_b4:NJ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	inputs_b3:NJ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	inputs_b2:N J
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	inputs_b1
  

O__inference_random_forest_model_layer_call_and_return_conditional_losses_105583
b1
b2
b3
b4
b5
b6
b7
evi
nbr
ndbi	
ndbai
ndmi
ndvi
ndwi
	elevation	
inference_op_model_handle
identity˘inference_opŇ
PartitionedCallPartitionedCallb1b2b3b4b5b6b7evinbrndbindbaindmindvindwi	elevation*
Tin
2	*
Tout
2*
_collective_manager_ids
 *÷
_output_shapesä
á:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *5
config_proto%#

CPU

GPU2*0J 8 J *3
f.R,
*__inference__build_normalized_inputs_19890ß
stackPackPartitionedCall:output:0PartitionedCall:output:1PartitionedCall:output:2PartitionedCall:output:3PartitionedCall:output:4PartitionedCall:output:5PartitionedCall:output:6PartitionedCall:output:7PartitionedCall:output:8PartitionedCall:output:9PartitionedCall:output:10PartitionedCall:output:11PartitionedCall:output:12PartitionedCall:output:13PartitionedCall:output:14*
N*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*

axisL
ConstConst*
_output_shapes
:  *
dtype0*
value
B  N
Const_1Const*
_output_shapes
:  *
dtype0*
value
B  X
RaggedConstant/valuesConst*
_output_shapes
: *
dtype0*
valueB ^
RaggedConstant/ConstConst*
_output_shapes
:*
dtype0	*
valueB	R `
RaggedConstant/Const_1Const*
_output_shapes
:*
dtype0	*
valueB	R Ą
inference_opSimpleMLInferenceOpWithHandlestack:output:0Const:output:0Const_1:output:0RaggedConstant/values:output:0RaggedConstant/Const:output:0RaggedConstant/Const_1:output:0inference_op_model_handle*-
_output_shapes
:˙˙˙˙˙˙˙˙˙
:
*
dense_output_dim
ŕ
PartitionedCall_1PartitionedCall inference_op:dense_predictions:0'inference_op:dense_col_representation:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *5
config_proto%#

CPU

GPU2*0J 8 J *0
f+R)
'__inference__finalize_predictions_19919i
IdentityIdentityPartitionedCall_1:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
1
NoOpNoOp^inference_op*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ř
_input_shapesć
ă:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: 2
inference_opinference_op:,(
&
_user_specified_namemodel_handle:NJ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	elevation:IE
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameNDWI:IE
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameNDVI:IE
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameNDMI:J
F
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameNDBaI:I	E
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameNDBI:HD
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameNBR:HD
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameEVI:GC
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameB7:GC
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameB6:GC
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameB5:GC
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameB4:GC
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameB3:GC
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameB2:G C
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameB1
´
Ŕ
__inference__initializer_105660
staticregexreplace_input>
:simple_ml_simplemlloadmodelfrompathwithhandle_model_handle
identity˘-simple_ml/SimpleMLLoadModelFromPathWithHandle
StaticRegexReplaceStaticRegexReplacestaticregexreplace_input*
_output_shapes
: *!
patterna7f2c213315f47eedone*
rewrite ć
-simple_ml/SimpleMLLoadModelFromPathWithHandle#SimpleMLLoadModelFromPathWithHandle:simple_ml_simplemlloadmodelfrompathwithhandle_model_handleStaticRegexReplace:output:0*
_output_shapes
 *!
file_prefixa7f2c213315f47eeG
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: R
NoOpNoOp.^simple_ml/SimpleMLLoadModelFromPathWithHandle*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 2^
-simple_ml/SimpleMLLoadModelFromPathWithHandle-simple_ml/SimpleMLLoadModelFromPathWithHandle:,(
&
_user_specified_namemodel_handle: 

_output_shapes
: 
Ü
ú
4__inference_random_forest_model_layer_call_fn_105604
b1
b2
b3
b4
b5
b6
b7
evi
nbr
ndbi	
ndbai
ndmi
ndvi
ndwi
	elevation	
unknown
identity˘StatefulPartitionedCall˛
StatefulPartitionedCallStatefulPartitionedCallb1b2b3b4b5b6b7evinbrndbindbaindmindvindwi	elevationunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *5
config_proto%#

CPU

GPU2*0J 8 J *X
fSRQ
O__inference_random_forest_model_layer_call_and_return_conditional_losses_105540o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ř
_input_shapesć
ă:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name105600:NJ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	elevation:IE
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameNDWI:IE
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameNDVI:IE
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameNDMI:J
F
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameNDBaI:I	E
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameNDBI:HD
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameNBR:HD
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameEVI:GC
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameB7:GC
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameB6:GC
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameB5:GC
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameB4:GC
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameB3:GC
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameB2:G C
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameB1
  

O__inference_random_forest_model_layer_call_and_return_conditional_losses_105540
b1
b2
b3
b4
b5
b6
b7
evi
nbr
ndbi	
ndbai
ndmi
ndvi
ndwi
	elevation	
inference_op_model_handle
identity˘inference_opŇ
PartitionedCallPartitionedCallb1b2b3b4b5b6b7evinbrndbindbaindmindvindwi	elevation*
Tin
2	*
Tout
2*
_collective_manager_ids
 *÷
_output_shapesä
á:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *5
config_proto%#

CPU

GPU2*0J 8 J *3
f.R,
*__inference__build_normalized_inputs_19890ß
stackPackPartitionedCall:output:0PartitionedCall:output:1PartitionedCall:output:2PartitionedCall:output:3PartitionedCall:output:4PartitionedCall:output:5PartitionedCall:output:6PartitionedCall:output:7PartitionedCall:output:8PartitionedCall:output:9PartitionedCall:output:10PartitionedCall:output:11PartitionedCall:output:12PartitionedCall:output:13PartitionedCall:output:14*
N*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*

axisL
ConstConst*
_output_shapes
:  *
dtype0*
value
B  N
Const_1Const*
_output_shapes
:  *
dtype0*
value
B  X
RaggedConstant/valuesConst*
_output_shapes
: *
dtype0*
valueB ^
RaggedConstant/ConstConst*
_output_shapes
:*
dtype0	*
valueB	R `
RaggedConstant/Const_1Const*
_output_shapes
:*
dtype0	*
valueB	R Ą
inference_opSimpleMLInferenceOpWithHandlestack:output:0Const:output:0Const_1:output:0RaggedConstant/values:output:0RaggedConstant/Const:output:0RaggedConstant/Const_1:output:0inference_op_model_handle*-
_output_shapes
:˙˙˙˙˙˙˙˙˙
:
*
dense_output_dim
ŕ
PartitionedCall_1PartitionedCall inference_op:dense_predictions:0'inference_op:dense_col_representation:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *5
config_proto%#

CPU

GPU2*0J 8 J *0
f+R)
'__inference__finalize_predictions_19919i
IdentityIdentityPartitionedCall_1:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
1
NoOpNoOp^inference_op*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ř
_input_shapesć
ă:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: 2
inference_opinference_op:,(
&
_user_specified_namemodel_handle:NJ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	elevation:IE
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameNDWI:IE
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameNDVI:IE
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameNDMI:J
F
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameNDBaI:I	E
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameNDBI:HD
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameNBR:HD
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameEVI:GC
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameB7:GC
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameB6:GC
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameB5:GC
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameB4:GC
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameB3:GC
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameB2:G C
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameB1
ô
ň
"__inference__traced_restore_105777
file_prefix%
assignvariableop_is_trained:
 &
assignvariableop_1_iteration:	 *
 assignvariableop_2_learning_rate: "
assignvariableop_3_total: "
assignvariableop_4_count: 

identity_6˘AssignVariableOp˘AssignVariableOp_1˘AssignVariableOp_2˘AssignVariableOp_3˘AssignVariableOp_4
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ž
value¤BĄB&_is_trained/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH|
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B B ź
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*,
_output_shapes
::::::*
dtypes

2
	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0
*
_output_shapes
:Ž
AssignVariableOpAssignVariableOpassignvariableop_is_trainedIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0
]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0	*
_output_shapes
:ł
AssignVariableOp_1AssignVariableOpassignvariableop_1_iterationIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:ˇ
AssignVariableOp_2AssignVariableOp assignvariableop_2_learning_rateIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Ż
AssignVariableOp_3AssignVariableOpassignvariableop_3_totalIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Ż
AssignVariableOp_4AssignVariableOpassignvariableop_4_countIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 Á

Identity_5Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^NoOp"/device:CPU:0*
T0*
_output_shapes
: U

Identity_6IdentityIdentity_5:output:0^NoOp_1*
T0*
_output_shapes
: 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4*
_output_shapes
 "!

identity_6Identity_6:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : : : : 2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42$
AssignVariableOpAssignVariableOp:%!

_user_specified_namecount:%!

_user_specified_nametotal:-)
'
_user_specified_namelearning_rate:)%
#
_user_specified_name	iteration:*&
$
_user_specified_name
is_trained:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
˘"
Ě
__inference_call_20213
	inputs_b1
	inputs_b2
	inputs_b3
	inputs_b4
	inputs_b5
	inputs_b6
	inputs_b7

inputs_evi

inputs_nbr
inputs_ndbi
inputs_ndbai
inputs_ndmi
inputs_ndvi
inputs_ndwi
inputs_elevation
inference_op_model_handle
identity˘inference_opť
PartitionedCallPartitionedCall	inputs_b1	inputs_b2	inputs_b3	inputs_b4	inputs_b5	inputs_b6	inputs_b7
inputs_evi
inputs_nbrinputs_ndbiinputs_ndbaiinputs_ndmiinputs_ndviinputs_ndwiinputs_elevation*
Tin
2*
Tout
2*
_collective_manager_ids
 *÷
_output_shapesä
á:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *5
config_proto%#

CPU

GPU2*0J 8 J *3
f.R,
*__inference__build_normalized_inputs_20122ß
stackPackPartitionedCall:output:0PartitionedCall:output:1PartitionedCall:output:2PartitionedCall:output:3PartitionedCall:output:4PartitionedCall:output:5PartitionedCall:output:6PartitionedCall:output:7PartitionedCall:output:8PartitionedCall:output:9PartitionedCall:output:10PartitionedCall:output:11PartitionedCall:output:12PartitionedCall:output:13PartitionedCall:output:14*
N*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*

axisL
ConstConst*
_output_shapes
:  *
dtype0*
value
B  N
Const_1Const*
_output_shapes
:  *
dtype0*
value
B  X
RaggedConstant/valuesConst*
_output_shapes
: *
dtype0*
valueB ^
RaggedConstant/ConstConst*
_output_shapes
:*
dtype0	*
valueB	R `
RaggedConstant/Const_1Const*
_output_shapes
:*
dtype0	*
valueB	R Ą
inference_opSimpleMLInferenceOpWithHandlestack:output:0Const:output:0Const_1:output:0RaggedConstant/values:output:0RaggedConstant/Const:output:0RaggedConstant/Const_1:output:0inference_op_model_handle*-
_output_shapes
:˙˙˙˙˙˙˙˙˙
:
*
dense_output_dim
ŕ
PartitionedCall_1PartitionedCall inference_op:dense_predictions:0'inference_op:dense_col_representation:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *5
config_proto%#

CPU

GPU2*0J 8 J *0
f+R)
'__inference__finalize_predictions_19919i
IdentityIdentityPartitionedCall_1:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
1
NoOpNoOp^inference_op*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ř
_input_shapesć
ă:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: 2
inference_opinference_op:,(
&
_user_specified_namemodel_handle:UQ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_user_specified_nameinputs_elevation:PL
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_nameinputs_ndwi:PL
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_nameinputs_ndvi:PL
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_nameinputs_ndmi:Q
M
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_nameinputs_ndbai:P	L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_nameinputs_ndbi:OK
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
inputs_nbr:OK
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
inputs_evi:NJ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	inputs_b7:NJ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	inputs_b6:NJ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	inputs_b5:NJ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	inputs_b4:NJ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	inputs_b3:NJ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	inputs_b2:N J
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	inputs_b1
Ż

'__inference__finalize_predictions_20127!
predictions_dense_predictions(
$predictions_dense_col_representation
identitye
IdentityIdentitypredictions_dense_predictions*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:˙˙˙˙˙˙˙˙˙
:
:`\

_output_shapes
:

>
_user_specified_name&$predictions_dense_col_representation:f b
'
_output_shapes
:˙˙˙˙˙˙˙˙˙

7
_user_specified_namepredictions_dense_predictions
Ž
L
__inference__creator_105653
identity˘SimpleMLCreateModelResource
SimpleMLCreateModelResourceSimpleMLCreateModelResource*
_output_shapes
: *E
shared_name64simple_ml_model_70553a36-9e93-4d8e-9413-5e8378ae7ed6h
IdentityIdentity*SimpleMLCreateModelResource:model_handle:0^NoOp*
T0*
_output_shapes
: @
NoOpNoOp^SimpleMLCreateModelResource*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2:
SimpleMLCreateModelResourceSimpleMLCreateModelResource
ř0
É
__inference__traced_save_105753
file_prefix+
!read_disablecopyonread_is_trained:
 ,
"read_1_disablecopyonread_iteration:	 0
&read_2_disablecopyonread_learning_rate: (
read_3_disablecopyonread_total: (
read_4_disablecopyonread_count: 
savev2_const
identity_11˘MergeV2Checkpoints˘Read/DisableCopyOnRead˘Read/ReadVariableOp˘Read_1/DisableCopyOnRead˘Read_1/ReadVariableOp˘Read_2/DisableCopyOnRead˘Read_2/ReadVariableOp˘Read_3/DisableCopyOnRead˘Read_3/ReadVariableOp˘Read_4/DisableCopyOnRead˘Read_4/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: d
Read/DisableCopyOnReadDisableCopyOnRead!read_disablecopyonread_is_trained*
_output_shapes
 
Read/ReadVariableOpReadVariableOp!read_disablecopyonread_is_trained^Read/DisableCopyOnRead*
_output_shapes
: *
dtype0
R
IdentityIdentityRead/ReadVariableOp:value:0*
T0
*
_output_shapes
: Y

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0
*
_output_shapes
: g
Read_1/DisableCopyOnReadDisableCopyOnRead"read_1_disablecopyonread_iteration*
_output_shapes
 
Read_1/ReadVariableOpReadVariableOp"read_1_disablecopyonread_iteration^Read_1/DisableCopyOnRead*
_output_shapes
: *
dtype0	V

Identity_2IdentityRead_1/ReadVariableOp:value:0*
T0	*
_output_shapes
: [

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0	*
_output_shapes
: k
Read_2/DisableCopyOnReadDisableCopyOnRead&read_2_disablecopyonread_learning_rate*
_output_shapes
 
Read_2/ReadVariableOpReadVariableOp&read_2_disablecopyonread_learning_rate^Read_2/DisableCopyOnRead*
_output_shapes
: *
dtype0V

Identity_4IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes
: [

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
: c
Read_3/DisableCopyOnReadDisableCopyOnReadread_3_disablecopyonread_total*
_output_shapes
 
Read_3/ReadVariableOpReadVariableOpread_3_disablecopyonread_total^Read_3/DisableCopyOnRead*
_output_shapes
: *
dtype0V

Identity_6IdentityRead_3/ReadVariableOp:value:0*
T0*
_output_shapes
: [

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
: c
Read_4/DisableCopyOnReadDisableCopyOnReadread_4_disablecopyonread_count*
_output_shapes
 
Read_4/ReadVariableOpReadVariableOpread_4_disablecopyonread_count^Read_4/DisableCopyOnRead*
_output_shapes
: *
dtype0V

Identity_8IdentityRead_4/ReadVariableOp:value:0*
T0*
_output_shapes
: [

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ž
value¤BĄB&_is_trained/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHy
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B B Ć
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtypes

2
	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:ł
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_10Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_11IdentityIdentity_10:output:0^NoOp*
T0*
_output_shapes
: ˛
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp*
_output_shapes
 "#
identity_11Identity_11:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp:=9

_output_shapes
: 

_user_specified_nameConst:%!

_user_specified_namecount:%!

_user_specified_nametotal:-)
'
_user_specified_namelearning_rate:)%
#
_user_specified_name	iteration:*&
$
_user_specified_name
is_trained:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ç

*__inference__build_normalized_inputs_20122
	inputs_b1
	inputs_b2
	inputs_b3
	inputs_b4
	inputs_b5
	inputs_b6
	inputs_b7

inputs_evi

inputs_nbr
inputs_ndbi
inputs_ndbai
inputs_ndmi
inputs_ndvi
inputs_ndwi
inputs_elevation
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7

identity_8

identity_9
identity_10
identity_11
identity_12
identity_13
identity_14M
IdentityIdentity	inputs_b1*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙O

Identity_1Identity	inputs_b2*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙O

Identity_2Identity	inputs_b3*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙O

Identity_3Identity	inputs_b4*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙O

Identity_4Identity	inputs_b5*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙O

Identity_5Identity	inputs_b6*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙O

Identity_6Identity	inputs_b7*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙P

Identity_7Identity
inputs_evi*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙P

Identity_8Identity
inputs_nbr*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙Q

Identity_9Identityinputs_ndbi*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙S
Identity_10Identityinputs_ndbai*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙R
Identity_11Identityinputs_ndmi*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙R
Identity_12Identityinputs_ndvi*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙R
Identity_13Identityinputs_ndwi*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙W
Identity_14Identityinputs_elevation*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙"#
identity_10Identity_10:output:0"#
identity_11Identity_11:output:0"#
identity_12Identity_12:output:0"#
identity_13Identity_13:output:0"#
identity_14Identity_14:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"!

identity_9Identity_9:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ö
_input_shapesä
á:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:UQ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_user_specified_nameinputs_elevation:PL
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_nameinputs_ndwi:PL
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_nameinputs_ndvi:PL
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_nameinputs_ndmi:Q
M
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_nameinputs_ndbai:P	L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_nameinputs_ndbi:OK
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
inputs_nbr:OK
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
inputs_evi:NJ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	inputs_b7:NJ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	inputs_b6:NJ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	inputs_b5:NJ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	inputs_b4:NJ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	inputs_b3:NJ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	inputs_b2:N J
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	inputs_b1

ę
$__inference_signature_wrapper_105649
b1
b2
b3
b4
b5
b6
b7
evi
nbr
ndbi	
ndbai
ndmi
ndvi
ndwi
	elevation	
unknown
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallb1b2b3b4b5b6b7evinbrndbindbaindmindvindwi	elevationunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *5
config_proto%#

CPU

GPU2*0J 8 J **
f%R#
!__inference__wrapped_model_105497o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ř
_input_shapesć
ă:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name105645:NJ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	elevation:IE
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameNDWI:IE
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameNDVI:IE
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameNDMI:J
F
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameNDBaI:I	E
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameNDBI:HD
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameNBR:HD
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameEVI:GC
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameB7:GC
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameB6:GC
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameB5:GC
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameB4:GC
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameB3:GC
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameB2:G C
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameB1

-
__inference__destroyer_105664
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
˘"
Ě
__inference_call_20170
	inputs_b1
	inputs_b2
	inputs_b3
	inputs_b4
	inputs_b5
	inputs_b6
	inputs_b7

inputs_evi

inputs_nbr
inputs_ndbi
inputs_ndbai
inputs_ndmi
inputs_ndvi
inputs_ndwi
inputs_elevation	
inference_op_model_handle
identity˘inference_opť
PartitionedCallPartitionedCall	inputs_b1	inputs_b2	inputs_b3	inputs_b4	inputs_b5	inputs_b6	inputs_b7
inputs_evi
inputs_nbrinputs_ndbiinputs_ndbaiinputs_ndmiinputs_ndviinputs_ndwiinputs_elevation*
Tin
2	*
Tout
2*
_collective_manager_ids
 *÷
_output_shapesä
á:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *5
config_proto%#

CPU

GPU2*0J 8 J *3
f.R,
*__inference__build_normalized_inputs_19890ß
stackPackPartitionedCall:output:0PartitionedCall:output:1PartitionedCall:output:2PartitionedCall:output:3PartitionedCall:output:4PartitionedCall:output:5PartitionedCall:output:6PartitionedCall:output:7PartitionedCall:output:8PartitionedCall:output:9PartitionedCall:output:10PartitionedCall:output:11PartitionedCall:output:12PartitionedCall:output:13PartitionedCall:output:14*
N*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*

axisL
ConstConst*
_output_shapes
:  *
dtype0*
value
B  N
Const_1Const*
_output_shapes
:  *
dtype0*
value
B  X
RaggedConstant/valuesConst*
_output_shapes
: *
dtype0*
valueB ^
RaggedConstant/ConstConst*
_output_shapes
:*
dtype0	*
valueB	R `
RaggedConstant/Const_1Const*
_output_shapes
:*
dtype0	*
valueB	R Ą
inference_opSimpleMLInferenceOpWithHandlestack:output:0Const:output:0Const_1:output:0RaggedConstant/values:output:0RaggedConstant/Const:output:0RaggedConstant/Const_1:output:0inference_op_model_handle*-
_output_shapes
:˙˙˙˙˙˙˙˙˙
:
*
dense_output_dim
ŕ
PartitionedCall_1PartitionedCall inference_op:dense_predictions:0'inference_op:dense_col_representation:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *5
config_proto%#

CPU

GPU2*0J 8 J *0
f+R)
'__inference__finalize_predictions_19919i
IdentityIdentityPartitionedCall_1:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
1
NoOpNoOp^inference_op*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ř
_input_shapesć
ă:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: 2
inference_opinference_op:,(
&
_user_specified_namemodel_handle:UQ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_user_specified_nameinputs_elevation:PL
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_nameinputs_ndwi:PL
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_nameinputs_ndvi:PL
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_nameinputs_ndmi:Q
M
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_nameinputs_ndbai:P	L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_nameinputs_ndbi:OK
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
inputs_nbr:OK
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
inputs_evi:NJ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	inputs_b7:NJ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	inputs_b6:NJ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	inputs_b5:NJ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	inputs_b4:NJ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	inputs_b3:NJ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	inputs_b2:N J
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	inputs_b1
Ą!
Ź
__inference_call_19922

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
	inputs_11
	inputs_12
inputs_9
	inputs_13
	inputs_10
	inputs_14	
inference_op_model_handle
identity˘inference_op
PartitionedCallPartitionedCallinputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8	inputs_11	inputs_12inputs_9	inputs_13	inputs_10	inputs_14*
Tin
2	*
Tout
2*
_collective_manager_ids
 *÷
_output_shapesä
á:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *5
config_proto%#

CPU

GPU2*0J 8 J *3
f.R,
*__inference__build_normalized_inputs_19890ß
stackPackPartitionedCall:output:0PartitionedCall:output:1PartitionedCall:output:2PartitionedCall:output:3PartitionedCall:output:4PartitionedCall:output:5PartitionedCall:output:6PartitionedCall:output:7PartitionedCall:output:8PartitionedCall:output:9PartitionedCall:output:10PartitionedCall:output:11PartitionedCall:output:12PartitionedCall:output:13PartitionedCall:output:14*
N*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*

axisL
ConstConst*
_output_shapes
:  *
dtype0*
value
B  N
Const_1Const*
_output_shapes
:  *
dtype0*
value
B  X
RaggedConstant/valuesConst*
_output_shapes
: *
dtype0*
valueB ^
RaggedConstant/ConstConst*
_output_shapes
:*
dtype0	*
valueB	R `
RaggedConstant/Const_1Const*
_output_shapes
:*
dtype0	*
valueB	R Ą
inference_opSimpleMLInferenceOpWithHandlestack:output:0Const:output:0Const_1:output:0RaggedConstant/values:output:0RaggedConstant/Const:output:0RaggedConstant/Const_1:output:0inference_op_model_handle*-
_output_shapes
:˙˙˙˙˙˙˙˙˙
:
*
dense_output_dim
ŕ
PartitionedCall_1PartitionedCall inference_op:dense_predictions:0'inference_op:dense_col_representation:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
* 
_read_only_resource_inputs
 *5
config_proto%#

CPU

GPU2*0J 8 J *0
f+R)
'__inference__finalize_predictions_19919i
IdentityIdentityPartitionedCall_1:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
1
NoOpNoOp^inference_op*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ř
_input_shapesć
ă:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: 2
inference_opinference_op:,(
&
_user_specified_namemodel_handle:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K
G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K	G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs"ĘL
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*×
serving_defaultĂ
-
B1'
serving_default_B1:0˙˙˙˙˙˙˙˙˙
-
B2'
serving_default_B2:0˙˙˙˙˙˙˙˙˙
-
B3'
serving_default_B3:0˙˙˙˙˙˙˙˙˙
-
B4'
serving_default_B4:0˙˙˙˙˙˙˙˙˙
-
B5'
serving_default_B5:0˙˙˙˙˙˙˙˙˙
-
B6'
serving_default_B6:0˙˙˙˙˙˙˙˙˙
-
B7'
serving_default_B7:0˙˙˙˙˙˙˙˙˙
/
EVI(
serving_default_EVI:0˙˙˙˙˙˙˙˙˙
/
NBR(
serving_default_NBR:0˙˙˙˙˙˙˙˙˙
1
NDBI)
serving_default_NDBI:0˙˙˙˙˙˙˙˙˙
3
NDBaI*
serving_default_NDBaI:0˙˙˙˙˙˙˙˙˙
1
NDMI)
serving_default_NDMI:0˙˙˙˙˙˙˙˙˙
1
NDVI)
serving_default_NDVI:0˙˙˙˙˙˙˙˙˙
1
NDWI)
serving_default_NDWI:0˙˙˙˙˙˙˙˙˙
;
	elevation.
serving_default_elevation:0	˙˙˙˙˙˙˙˙˙<
output_10
StatefulPartitionedCall:0˙˙˙˙˙˙˙˙˙
tensorflow/serving/predict22

asset_path_initializer:0a7f2c213315f47eedone2D

asset_path_initializer_1:0$a7f2c213315f47eenodes-00000-of-000012G

asset_path_initializer_2:0'a7f2c213315f47eerandom_forest_header.pb2<

asset_path_initializer_3:0a7f2c213315f47eedata_spec.pb29

asset_path_initializer_4:0a7f2c213315f47eeheader.pb:ô
ś
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

_multitask
	_is_trained

_learner_params
	_features
	optimizer
loss
_models
_build_normalized_inputs
_finalize_predictions
call
call_get_leaves
yggdrasil_model_path_tensor

signatures"
_tf_keras_model
'
	0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ę
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ď
trace_0
trace_12
4__inference_random_forest_model_layer_call_fn_105604
4__inference_random_forest_model_layer_call_fn_105625Š
˘˛
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaults˘
p 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 ztrace_0ztrace_1

trace_0
trace_12Î
O__inference_random_forest_model_layer_call_and_return_conditional_losses_105540
O__inference_random_forest_model_layer_call_and_return_conditional_losses_105583Š
˘˛
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaults˘
p 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 ztrace_0ztrace_1
B
!__inference__wrapped_model_105497B1B2B3B4B5B6B7EVINBRNDBINDBaINDMINDVINDWI	elevation"
˛
FullArgSpec
args

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
 "
trackable_list_wrapper
:
 2
is_trained
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
j

_variables
_iterations
 _learning_rate
!_update_step_xla"
experimentalOptimizer
 "
trackable_dict_wrapper
'
"0"
trackable_list_wrapper
Ş
#trace_0
$trace_12ó
*__inference__build_normalized_inputs_20090
*__inference__build_normalized_inputs_20122
˛
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 z#trace_0z$trace_1

%trace_02ĺ
'__inference__finalize_predictions_20127š
˛˛Ž
FullArgSpec1
args)&
jtask
jpredictions
jlike_engine
varargs
 
varkw
 
defaults˘
p 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 z%trace_0

&trace_0
'trace_12Ü
__inference_call_20170
__inference_call_20213Š
˘˛
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaults˘
p 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 z&trace_0z'trace_1
2
˛
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ü
(trace_02ß
-__inference_yggdrasil_model_path_tensor_20218­
Ľ˛Ą
FullArgSpec$
args
jmultitask_model_index
varargs
 
varkw
 
defaults˘
` 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ z(trace_0
,
)serving_default"
signature_map
'
	0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
*0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
˛BŻ
4__inference_random_forest_model_layer_call_fn_105604B1B2B3B4B5B6B7EVINBRNDBINDBaINDMINDVINDWI	elevation"¤
˛
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
˛BŻ
4__inference_random_forest_model_layer_call_fn_105625B1B2B3B4B5B6B7EVINBRNDBINDBaINDMINDVINDWI	elevation"¤
˛
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ÍBĘ
O__inference_random_forest_model_layer_call_and_return_conditional_losses_105540B1B2B3B4B5B6B7EVINBRNDBINDBaINDMINDVINDWI	elevation"¤
˛
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ÍBĘ
O__inference_random_forest_model_layer_call_and_return_conditional_losses_105583B1B2B3B4B5B6B7EVINBRNDBINDBaINDMINDVINDWI	elevation"¤
˛
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
'
0"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
ľ2˛Ż
Ś˛˘
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 0
G
+_input_builder
,_compiled_model"
_generic_user_object
B
*__inference__build_normalized_inputs_20090	inputs_b1	inputs_b2	inputs_b3	inputs_b4	inputs_b5	inputs_b6	inputs_b7
inputs_evi
inputs_nbrinputs_ndbiinputs_ndbaiinputs_ndmiinputs_ndviinputs_ndwiinputs_elevation"
˛
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
B
*__inference__build_normalized_inputs_20122	inputs_b1	inputs_b2	inputs_b3	inputs_b4	inputs_b5	inputs_b6	inputs_b7
inputs_evi
inputs_nbrinputs_ndbiinputs_ndbaiinputs_ndmiinputs_ndviinputs_ndwiinputs_elevation"
˛
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ŞB§
'__inference__finalize_predictions_20127predictions_dense_predictions$predictions_dense_col_representation"´
­˛Š
FullArgSpec1
args)&
jtask
jpredictions
jlike_engine
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ýBú
__inference_call_20170	inputs_b1	inputs_b2	inputs_b3	inputs_b4	inputs_b5	inputs_b6	inputs_b7
inputs_evi
inputs_nbrinputs_ndbiinputs_ndbaiinputs_ndmiinputs_ndviinputs_ndwiinputs_elevation"¤
˛
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ýBú
__inference_call_20213	inputs_b1	inputs_b2	inputs_b3	inputs_b4	inputs_b5	inputs_b6	inputs_b7
inputs_evi
inputs_nbrinputs_ndbiinputs_ndbaiinputs_ndmiinputs_ndviinputs_ndwiinputs_elevation"¤
˛
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ú
-	capture_0BŮ
-__inference_yggdrasil_model_path_tensor_20218"§
 ˛
FullArgSpec$
args
jmultitask_model_index
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 z-	capture_0
řBő
$__inference_signature_wrapper_105649B1B2B3B4B5B6B7EVINBRNDBINDBaINDMINDVINDWI	elevation"ü
ő˛ń
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargsqn
jB1
jB2
jB3
jB4
jB5
jB6
jB7
jEVI
jNBR
jNDBI
jNDBaI
jNDMI
jNDVI
jNDWI
j	elevation
kwonlydefaults
 
annotationsŞ *
 
N
.	variables
/	keras_api
	0total
	1count"
_tf_keras_metric
l
2_feature_name_to_idx
3	_init_ops
#4categorical_str_to_int_hashmaps"
_generic_user_object
S
5_model_loader
6_create_resource
7_initialize
8_destroy_resourceR 
* 
.
00
11"
trackable_list_wrapper
-
.	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
Q
9_output_types
:
_all_files
-
_done_file"
_generic_user_object
Ě
;trace_02Ż
__inference__creator_105653
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ z;trace_0
Đ
<trace_02ł
__inference__initializer_105660
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ z<trace_0
Î
=trace_02ą
__inference__destroyer_105664
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ z=trace_0
 "
trackable_list_wrapper
C
>0
?1
-2
@3
A4"
trackable_list_wrapper
˛BŻ
__inference__creator_105653"
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ 
Ô
-	capture_0Bł
__inference__initializer_105660"
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ z-	capture_0
´Bą
__inference__destroyer_105664"
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ 
*
*
*
*É	
*__inference__build_normalized_inputs_20090	˘
ü˘ř
őŞń
%
B1
	inputs_b1˙˙˙˙˙˙˙˙˙
%
B2
	inputs_b2˙˙˙˙˙˙˙˙˙
%
B3
	inputs_b3˙˙˙˙˙˙˙˙˙
%
B4
	inputs_b4˙˙˙˙˙˙˙˙˙
%
B5
	inputs_b5˙˙˙˙˙˙˙˙˙
%
B6
	inputs_b6˙˙˙˙˙˙˙˙˙
%
B7
	inputs_b7˙˙˙˙˙˙˙˙˙
'
EVI 

inputs_evi˙˙˙˙˙˙˙˙˙
'
NBR 

inputs_nbr˙˙˙˙˙˙˙˙˙
)
NDBI!
inputs_ndbi˙˙˙˙˙˙˙˙˙
+
NDBaI"
inputs_ndbai˙˙˙˙˙˙˙˙˙
)
NDMI!
inputs_ndmi˙˙˙˙˙˙˙˙˙
)
NDVI!
inputs_ndvi˙˙˙˙˙˙˙˙˙
)
NDWI!
inputs_ndwi˙˙˙˙˙˙˙˙˙
3
	elevation&#
inputs_elevation˙˙˙˙˙˙˙˙˙	
Ş "Ş

B1
b1˙˙˙˙˙˙˙˙˙

B2
b2˙˙˙˙˙˙˙˙˙

B3
b3˙˙˙˙˙˙˙˙˙

B4
b4˙˙˙˙˙˙˙˙˙

B5
b5˙˙˙˙˙˙˙˙˙

B6
b6˙˙˙˙˙˙˙˙˙

B7
b7˙˙˙˙˙˙˙˙˙
 
EVI
evi˙˙˙˙˙˙˙˙˙
 
NBR
nbr˙˙˙˙˙˙˙˙˙
"
NDBI
ndbi˙˙˙˙˙˙˙˙˙
$
NDBaI
ndbai˙˙˙˙˙˙˙˙˙
"
NDMI
ndmi˙˙˙˙˙˙˙˙˙
"
NDVI
ndvi˙˙˙˙˙˙˙˙˙
"
NDWI
ndwi˙˙˙˙˙˙˙˙˙
,
	elevation
	elevation˙˙˙˙˙˙˙˙˙É	
*__inference__build_normalized_inputs_20122	˘
ü˘ř
őŞń
%
B1
	inputs_b1˙˙˙˙˙˙˙˙˙
%
B2
	inputs_b2˙˙˙˙˙˙˙˙˙
%
B3
	inputs_b3˙˙˙˙˙˙˙˙˙
%
B4
	inputs_b4˙˙˙˙˙˙˙˙˙
%
B5
	inputs_b5˙˙˙˙˙˙˙˙˙
%
B6
	inputs_b6˙˙˙˙˙˙˙˙˙
%
B7
	inputs_b7˙˙˙˙˙˙˙˙˙
'
EVI 

inputs_evi˙˙˙˙˙˙˙˙˙
'
NBR 

inputs_nbr˙˙˙˙˙˙˙˙˙
)
NDBI!
inputs_ndbi˙˙˙˙˙˙˙˙˙
+
NDBaI"
inputs_ndbai˙˙˙˙˙˙˙˙˙
)
NDMI!
inputs_ndmi˙˙˙˙˙˙˙˙˙
)
NDVI!
inputs_ndvi˙˙˙˙˙˙˙˙˙
)
NDWI!
inputs_ndwi˙˙˙˙˙˙˙˙˙
3
	elevation&#
inputs_elevation˙˙˙˙˙˙˙˙˙
Ş "Ş

B1
b1˙˙˙˙˙˙˙˙˙

B2
b2˙˙˙˙˙˙˙˙˙

B3
b3˙˙˙˙˙˙˙˙˙

B4
b4˙˙˙˙˙˙˙˙˙

B5
b5˙˙˙˙˙˙˙˙˙

B6
b6˙˙˙˙˙˙˙˙˙

B7
b7˙˙˙˙˙˙˙˙˙
 
EVI
evi˙˙˙˙˙˙˙˙˙
 
NBR
nbr˙˙˙˙˙˙˙˙˙
"
NDBI
ndbi˙˙˙˙˙˙˙˙˙
$
NDBaI
ndbai˙˙˙˙˙˙˙˙˙
"
NDMI
ndmi˙˙˙˙˙˙˙˙˙
"
NDVI
ndvi˙˙˙˙˙˙˙˙˙
"
NDWI
ndwi˙˙˙˙˙˙˙˙˙
,
	elevation
	elevation˙˙˙˙˙˙˙˙˙@
__inference__creator_105653!˘

˘ 
Ş "
unknown B
__inference__destroyer_105664!˘

˘ 
Ş "
unknown 
'__inference__finalize_predictions_20127ďÉ˘Ĺ
˝˘š
`
Ž˛Ş
ModelOutputL
dense_predictions74
predictions_dense_predictions˙˙˙˙˙˙˙˙˙
M
dense_col_representation1.
$predictions_dense_col_representation

p 
Ş "!
unknown˙˙˙˙˙˙˙˙˙
H
__inference__initializer_105660%-,˘

˘ 
Ş "
unknown 
!__inference__wrapped_model_105497Ú,˘
˘
Ş

B1
B1˙˙˙˙˙˙˙˙˙

B2
B2˙˙˙˙˙˙˙˙˙

B3
B3˙˙˙˙˙˙˙˙˙

B4
B4˙˙˙˙˙˙˙˙˙

B5
B5˙˙˙˙˙˙˙˙˙

B6
B6˙˙˙˙˙˙˙˙˙

B7
B7˙˙˙˙˙˙˙˙˙
 
EVI
EVI˙˙˙˙˙˙˙˙˙
 
NBR
NBR˙˙˙˙˙˙˙˙˙
"
NDBI
NDBI˙˙˙˙˙˙˙˙˙
$
NDBaI
NDBaI˙˙˙˙˙˙˙˙˙
"
NDMI
NDMI˙˙˙˙˙˙˙˙˙
"
NDVI
NDVI˙˙˙˙˙˙˙˙˙
"
NDWI
NDWI˙˙˙˙˙˙˙˙˙
,
	elevation
	elevation˙˙˙˙˙˙˙˙˙	
Ş "3Ş0
.
output_1"
output_1˙˙˙˙˙˙˙˙˙
Đ
__inference_call_20170ľ,˘
˘ü
őŞń
%
B1
	inputs_b1˙˙˙˙˙˙˙˙˙
%
B2
	inputs_b2˙˙˙˙˙˙˙˙˙
%
B3
	inputs_b3˙˙˙˙˙˙˙˙˙
%
B4
	inputs_b4˙˙˙˙˙˙˙˙˙
%
B5
	inputs_b5˙˙˙˙˙˙˙˙˙
%
B6
	inputs_b6˙˙˙˙˙˙˙˙˙
%
B7
	inputs_b7˙˙˙˙˙˙˙˙˙
'
EVI 

inputs_evi˙˙˙˙˙˙˙˙˙
'
NBR 

inputs_nbr˙˙˙˙˙˙˙˙˙
)
NDBI!
inputs_ndbi˙˙˙˙˙˙˙˙˙
+
NDBaI"
inputs_ndbai˙˙˙˙˙˙˙˙˙
)
NDMI!
inputs_ndmi˙˙˙˙˙˙˙˙˙
)
NDVI!
inputs_ndvi˙˙˙˙˙˙˙˙˙
)
NDWI!
inputs_ndwi˙˙˙˙˙˙˙˙˙
3
	elevation&#
inputs_elevation˙˙˙˙˙˙˙˙˙	
p 
Ş "!
unknown˙˙˙˙˙˙˙˙˙
Đ
__inference_call_20213ľ,˘
˘ü
őŞń
%
B1
	inputs_b1˙˙˙˙˙˙˙˙˙
%
B2
	inputs_b2˙˙˙˙˙˙˙˙˙
%
B3
	inputs_b3˙˙˙˙˙˙˙˙˙
%
B4
	inputs_b4˙˙˙˙˙˙˙˙˙
%
B5
	inputs_b5˙˙˙˙˙˙˙˙˙
%
B6
	inputs_b6˙˙˙˙˙˙˙˙˙
%
B7
	inputs_b7˙˙˙˙˙˙˙˙˙
'
EVI 

inputs_evi˙˙˙˙˙˙˙˙˙
'
NBR 

inputs_nbr˙˙˙˙˙˙˙˙˙
)
NDBI!
inputs_ndbi˙˙˙˙˙˙˙˙˙
+
NDBaI"
inputs_ndbai˙˙˙˙˙˙˙˙˙
)
NDMI!
inputs_ndmi˙˙˙˙˙˙˙˙˙
)
NDVI!
inputs_ndvi˙˙˙˙˙˙˙˙˙
)
NDWI!
inputs_ndwi˙˙˙˙˙˙˙˙˙
3
	elevation&#
inputs_elevation˙˙˙˙˙˙˙˙˙
p 
Ş "!
unknown˙˙˙˙˙˙˙˙˙
Ť
O__inference_random_forest_model_layer_call_and_return_conditional_losses_105540×,Ł˘
˘
Ş

B1
B1˙˙˙˙˙˙˙˙˙

B2
B2˙˙˙˙˙˙˙˙˙

B3
B3˙˙˙˙˙˙˙˙˙

B4
B4˙˙˙˙˙˙˙˙˙

B5
B5˙˙˙˙˙˙˙˙˙

B6
B6˙˙˙˙˙˙˙˙˙

B7
B7˙˙˙˙˙˙˙˙˙
 
EVI
EVI˙˙˙˙˙˙˙˙˙
 
NBR
NBR˙˙˙˙˙˙˙˙˙
"
NDBI
NDBI˙˙˙˙˙˙˙˙˙
$
NDBaI
NDBaI˙˙˙˙˙˙˙˙˙
"
NDMI
NDMI˙˙˙˙˙˙˙˙˙
"
NDVI
NDVI˙˙˙˙˙˙˙˙˙
"
NDWI
NDWI˙˙˙˙˙˙˙˙˙
,
	elevation
	elevation˙˙˙˙˙˙˙˙˙	
p
Ş ",˘)
"
tensor_0˙˙˙˙˙˙˙˙˙

 Ť
O__inference_random_forest_model_layer_call_and_return_conditional_losses_105583×,Ł˘
˘
Ş

B1
B1˙˙˙˙˙˙˙˙˙

B2
B2˙˙˙˙˙˙˙˙˙

B3
B3˙˙˙˙˙˙˙˙˙

B4
B4˙˙˙˙˙˙˙˙˙

B5
B5˙˙˙˙˙˙˙˙˙

B6
B6˙˙˙˙˙˙˙˙˙

B7
B7˙˙˙˙˙˙˙˙˙
 
EVI
EVI˙˙˙˙˙˙˙˙˙
 
NBR
NBR˙˙˙˙˙˙˙˙˙
"
NDBI
NDBI˙˙˙˙˙˙˙˙˙
$
NDBaI
NDBaI˙˙˙˙˙˙˙˙˙
"
NDMI
NDMI˙˙˙˙˙˙˙˙˙
"
NDVI
NDVI˙˙˙˙˙˙˙˙˙
"
NDWI
NDWI˙˙˙˙˙˙˙˙˙
,
	elevation
	elevation˙˙˙˙˙˙˙˙˙	
p 
Ş ",˘)
"
tensor_0˙˙˙˙˙˙˙˙˙

 
4__inference_random_forest_model_layer_call_fn_105604Ě,Ł˘
˘
Ş

B1
B1˙˙˙˙˙˙˙˙˙

B2
B2˙˙˙˙˙˙˙˙˙

B3
B3˙˙˙˙˙˙˙˙˙

B4
B4˙˙˙˙˙˙˙˙˙

B5
B5˙˙˙˙˙˙˙˙˙

B6
B6˙˙˙˙˙˙˙˙˙

B7
B7˙˙˙˙˙˙˙˙˙
 
EVI
EVI˙˙˙˙˙˙˙˙˙
 
NBR
NBR˙˙˙˙˙˙˙˙˙
"
NDBI
NDBI˙˙˙˙˙˙˙˙˙
$
NDBaI
NDBaI˙˙˙˙˙˙˙˙˙
"
NDMI
NDMI˙˙˙˙˙˙˙˙˙
"
NDVI
NDVI˙˙˙˙˙˙˙˙˙
"
NDWI
NDWI˙˙˙˙˙˙˙˙˙
,
	elevation
	elevation˙˙˙˙˙˙˙˙˙	
p
Ş "!
unknown˙˙˙˙˙˙˙˙˙

4__inference_random_forest_model_layer_call_fn_105625Ě,Ł˘
˘
Ş

B1
B1˙˙˙˙˙˙˙˙˙

B2
B2˙˙˙˙˙˙˙˙˙

B3
B3˙˙˙˙˙˙˙˙˙

B4
B4˙˙˙˙˙˙˙˙˙

B5
B5˙˙˙˙˙˙˙˙˙

B6
B6˙˙˙˙˙˙˙˙˙

B7
B7˙˙˙˙˙˙˙˙˙
 
EVI
EVI˙˙˙˙˙˙˙˙˙
 
NBR
NBR˙˙˙˙˙˙˙˙˙
"
NDBI
NDBI˙˙˙˙˙˙˙˙˙
$
NDBaI
NDBaI˙˙˙˙˙˙˙˙˙
"
NDMI
NDMI˙˙˙˙˙˙˙˙˙
"
NDVI
NDVI˙˙˙˙˙˙˙˙˙
"
NDWI
NDWI˙˙˙˙˙˙˙˙˙
,
	elevation
	elevation˙˙˙˙˙˙˙˙˙	
p 
Ş "!
unknown˙˙˙˙˙˙˙˙˙
ü
$__inference_signature_wrapper_105649Ó,˘
˘ 
Ş

B1
b1˙˙˙˙˙˙˙˙˙

B2
b2˙˙˙˙˙˙˙˙˙

B3
b3˙˙˙˙˙˙˙˙˙

B4
b4˙˙˙˙˙˙˙˙˙

B5
b5˙˙˙˙˙˙˙˙˙

B6
b6˙˙˙˙˙˙˙˙˙

B7
b7˙˙˙˙˙˙˙˙˙
 
EVI
evi˙˙˙˙˙˙˙˙˙
 
NBR
nbr˙˙˙˙˙˙˙˙˙
"
NDBI
ndbi˙˙˙˙˙˙˙˙˙
$
NDBaI
ndbai˙˙˙˙˙˙˙˙˙
"
NDMI
ndmi˙˙˙˙˙˙˙˙˙
"
NDVI
ndvi˙˙˙˙˙˙˙˙˙
"
NDWI
ndwi˙˙˙˙˙˙˙˙˙
,
	elevation
	elevation˙˙˙˙˙˙˙˙˙	"3Ş0
.
output_1"
output_1˙˙˙˙˙˙˙˙˙
Y
-__inference_yggdrasil_model_path_tensor_20218(-˘
˘
` 
Ş "
unknown 