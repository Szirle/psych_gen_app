import 'package:flutter/material.dart';
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
      home: TestApiWidget(),
      // home: const MyHomePage(title: 'Flutt`er Demo Home Page'),
    );
  }
}

class TestApiWidget extends StatefulWidget {
  @override
  _TestApiWidgetState createState() => _TestApiWidgetState();
}

class _TestApiWidgetState extends State<TestApiWidget> {
  String _message = 'Fetching data...';

  @override
  void initState() {
    super.initState();
    _fetchData();
  }

  Future<void> _fetchData() async {
    final response = await http.get(Uri.parse('/test'));  // Using a relative URL

    if (response.statusCode == 200) {
      setState(() {
        _message = json.decode(response.body)['message'];
      });
    } else {
      setState(() {
        _message = 'Failed to load data!';
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Text(_message);
  }
}

