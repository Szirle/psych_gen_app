import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:psych_gen_app/bloc/face_manipulation_bloc.dart';
import 'package:psych_gen_app/home_page.dart';
import 'package:psych_gen_app/repository/face_manipulation_repository_impl.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        fontFamily: 'WorkSans',
        scaffoldBackgroundColor: Colors.grey.shade50,
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.black),
        useMaterial3: true,
      ),
      home: BlocProvider(
        create: (context) => FaceManipulationBloc(
          repository: FaceManipulationRepositoryImpl(),
        ),
        child: const MyHomePage(title: 'Flutter Demo Home Page'),
      ),
    );
  }
}
