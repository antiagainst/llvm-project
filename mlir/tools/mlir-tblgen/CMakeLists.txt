set(LLVM_LINK_COMPONENTS
  MLIRTableGen
  Support
  )

add_tablegen(mlir-tblgen MLIR
  EnumsGen.cpp
  LLVMIRConversionGen.cpp
  LLVMIRIntrinsicGen.cpp
  mlir-tblgen.cpp
  OpDefinitionsGen.cpp
  OpDocGen.cpp
  OpInterfacesGen.cpp
  ReferenceImplGen.cpp
  RewriterGen.cpp
  SPIRVUtilsGen.cpp
  StructsGen.cpp
  )
target_link_libraries(mlir-tblgen PRIVATE MLIRSupport)
set_target_properties(mlir-tblgen PROPERTIES FOLDER "Tablegenning")
