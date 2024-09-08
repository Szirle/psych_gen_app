import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:psych_gen_app/api/api_service.dart';
import 'package:psych_gen_app/home_page.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

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
      home: const MyHomePage(title: 'Flutter Demo Home Page'),
    );
  }
}

class TestApiWidget extends StatefulWidget {
  const TestApiWidget({Key? key}) : super(key: key);

  @override
  _TestApiWidgetState createState() => _TestApiWidgetState();
}

class _TestApiWidgetState extends State<TestApiWidget> {
  Uint8List? _imageBytes;
  bool _isLoading = true; // Track loading status

  @override
  void initState() {
    super.initState();
    _fetchImage();
  }

  Future<void> _fetchImage() async {
    ApiService apiService = ApiService();
    try {
      Uint8List imageBytes = await apiService.fetchImage();
      setState(() {
        _imageBytes = imageBytes;
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _isLoading = false;
      });
      print('Error fetching image: $e');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('API Image Fetch Example'),
      ),
      body: Center(
        child: _isLoading
            ? const CircularProgressIndicator()
            : _imageBytes != null
                ? Image.memory(_imageBytes!)
                : const Text('Failed to load image'),
      ),
    );
  }
}
