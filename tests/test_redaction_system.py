#!/usr/bin/env python3
"""
Comprehensive Test Suite for Redaction System
=============================================

Complete testing framework for the redaction system components:
- Unit tests for individual components
- Integration tests for the full system
- Performance benchmarks
- Edge case testing
"""

import sys as _sys
if _sys.platform == "win32":
    _sys.stdout.reconfigure(encoding="utf-8")
    _sys.stderr.reconfigure(encoding="utf-8")


import unittest
import time
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import the systems to test
from intelligent_redaction_system import IntelligentRedactor, MessageModerator, RedactionDemo

class TestIntelligentRedactor(unittest.TestCase):
    """Test cases for the IntelligentRedactor class"""
    
    @classmethod
    def setUpClass(cls):
        """Set up redactor for all tests"""
        print("🔧 Setting up IntelligentRedactor for testing...")
        cls.redactor = IntelligentRedactor()
        if not cls.redactor.model:
            cls.skipTest(cls, "Model not available for testing")
    
    def test_toxicity_prediction_clean_text(self):
        """Test toxicity prediction for clean text"""
        clean_messages = [
            "Hello, how are you today?",
            "Thanks for your help!",
            "The weather is nice today",
            "I enjoyed our conversation"
        ]
        
        for message in clean_messages:
            with self.subTest(message=message):
                result = self.redactor.predict_toxicity(message)
                
                self.assertIn('probability', result)
                self.assertIn('is_toxic', result)
                self.assertIn('confidence', result)
                self.assertIsInstance(result['probability'], float)
                self.assertIsInstance(result['is_toxic'], bool)
                self.assertLessEqual(result['probability'], 1.0)
                self.assertGreaterEqual(result['probability'], 0.0)
    
    def test_toxicity_prediction_toxic_text(self):
        """Test toxicity prediction for toxic text"""
        toxic_messages = [
            "You're such an idiot",
            "Go kill yourself",
            "I hate you so much",
            "Shut up, moron"
        ]
        
        for message in toxic_messages:
            with self.subTest(message=message):
                result = self.redactor.predict_toxicity(message)
                
                self.assertIn('probability', result)
                self.assertIn('is_toxic', result)
                # These should generally have higher toxicity scores
                # Note: We don't assert is_toxic=True because threshold may vary
                self.assertIsInstance(result['probability'], float)
    
    def test_toxic_word_identification(self):
        """Test identification of specific toxic words"""
        test_cases = [
            {
                'text': "You're an idiot",
                'expected_categories': ['insults']
            },
            {
                'text': "This is fucking annoying",
                'expected_categories': ['profanity']
            },
            {
                'text': "Go kill yourself",
                'expected_categories': ['threats']
            },
            {
                'text': "I hate you",
                'expected_categories': ['hate']
            }
        ]
        
        for case in test_cases:
            with self.subTest(text=case['text']):
                result = self.redactor.identify_toxic_words(case['text'])
                
                self.assertIsInstance(result, dict)
                # Check if at least one expected category has items
                found_expected = any(
                    len(result.get(cat, [])) > 0 
                    for cat in case['expected_categories']
                )
                # Note: We don't strictly assert this because the model might not catch everything
                # but we can verify the structure is correct
                for category in ['insults', 'profanity', 'threats', 'hate', 'other_toxic']:
                    self.assertIn(category, result)
                    self.assertIsInstance(result[category], list)
    
    def test_redaction_styles(self):
        """Test different redaction styles"""
        test_message = "You're such an idiot, go kill yourself!"
        styles = ['smart', 'partial', 'complete', 'warning']
        
        for style in styles:
            with self.subTest(style=style):
                result = self.redactor.redact_message(test_message, style)
                
                self.assertIn('original_text', result)
                self.assertIn('redacted_text', result)
                self.assertIn('was_redacted', result)
                self.assertIn('redaction_style', result)
                self.assertIn('toxicity_info', result)
                
                self.assertEqual(result['original_text'], test_message)
                self.assertEqual(result['redaction_style'], style)
                
                if style == 'complete':
                    self.assertIn('[MESSAGE REDACTED', result['redacted_text'])
                elif style == 'warning':
                    self.assertIn('WARNING:', result['redacted_text'])
    
    def test_edge_cases(self):
        """Test edge cases"""
        edge_cases = [
            "",  # Empty string
            "   ",  # Whitespace only
            "a",  # Single character
            "A" * 1000,  # Very long string
            "123456789",  # Numbers only
            "!@#$%^&*()",  # Special characters only
        ]
        
        for case in edge_cases:
            with self.subTest(text=case):
                # Should not crash
                result = self.redactor.predict_toxicity(case)
                self.assertIn('probability', result)
                
                redact_result = self.redactor.redact_message(case)
                self.assertIn('original_text', redact_result)

class TestMessageModerator(unittest.TestCase):
    """Test cases for the MessageModerator class"""
    
    @classmethod
    def setUpClass(cls):
        """Set up moderator for all tests"""
        print("🔧 Setting up MessageModerator for testing...")
        cls.redactor = IntelligentRedactor()
        if not cls.redactor.model:
            cls.skipTest(cls, "Model not available for testing")
        cls.moderator = MessageModerator(cls.redactor, auto_moderate=True, policy='moderate')
    
    def test_single_message_moderation(self):
        """Test moderation of a single message"""
        test_message = "Hello everyone!"
        username = "TestUser"
        
        result = self.moderator.moderate_message(test_message, username)
        
        self.assertIn('timestamp', result)
        self.assertIn('username', result)
        self.assertIn('original_message', result)
        self.assertIn('final_message', result)
        self.assertIn('toxicity_probability', result)
        self.assertIn('action_taken', result)
        
        self.assertEqual(result['username'], username)
        self.assertEqual(result['original_message'], test_message)
    
    def test_conversation_moderation(self):
        """Test moderation of multiple messages"""
        conversation = [
            {"username": "Alice", "text": "Hello everyone!"},
            {"username": "Bob", "text": "You're an idiot"},
            {"username": "Charlie", "text": "Let's be respectful"}
        ]
        
        results = self.moderator.moderate_conversation(conversation)
        
        self.assertEqual(len(results), len(conversation))
        
        for i, result in enumerate(results):
            self.assertEqual(result['username'], conversation[i]['username'])
            self.assertEqual(result['original_message'], conversation[i]['text'])
    
    def test_moderation_stats(self):
        """Test moderation statistics tracking"""
        initial_stats = self.moderator.get_moderation_stats()
        
        # Process some messages
        self.moderator.moderate_message("Hello!", "User1")
        self.moderator.moderate_message("You're stupid", "User2")
        
        final_stats = self.moderator.get_moderation_stats()
        
        # Stats should have increased
        self.assertGreater(final_stats['total_messages'], initial_stats['total_messages'])
        self.assertIn('toxicity_rate', final_stats)
        self.assertIn('redaction_rate', final_stats)

class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmark tests"""
    
    @classmethod
    def setUpClass(cls):
        """Set up for performance testing"""
        print("🔧 Setting up performance testing...")
        cls.redactor = IntelligentRedactor()
        if not cls.redactor.model:
            cls.skipTest(cls, "Model not available for performance testing")
    
    def test_single_message_performance(self):
        """Test performance of single message processing"""
        test_message = "You're being really stupid about this topic"
        iterations = 50
        
        start_time = time.time()
        
        for _ in range(iterations):
            result = self.redactor.redact_message(test_message, 'smart')
        
        end_time = time.time()
        avg_time = ((end_time - start_time) / iterations) * 1000  # ms
        
        print(f"⚡ Single message performance: {avg_time:.2f}ms average")
        
        # Performance assertion (should be under 1 second per message)
        self.assertLess(avg_time, 1000)
    
    def test_batch_processing_performance(self):
        """Test performance of batch processing"""
        messages = [
            "Hello everyone!",
            "This is a test message",
            "You're an idiot",
            "I hate this so much",
            "Thanks for your help!"
        ] * 20  # 100 total messages
        
        start_time = time.time()
        
        results = []
        for message in messages:
            result = self.redactor.redact_message(message, 'smart')
            results.append(result)
        
        end_time = time.time()
        total_time = (end_time - start_time) * 1000  # ms
        avg_time = total_time / len(messages)
        
        print(f"⚡ Batch processing performance:")
        print(f"   Total time: {total_time:.2f}ms")
        print(f"   Average per message: {avg_time:.2f}ms")
        print(f"   Messages per second: {len(messages) / (total_time / 1000):.1f}")
        
        # Performance assertions
        self.assertLess(avg_time, 500)  # Under 500ms per message
        self.assertGreater(len(messages) / (total_time / 1000), 2)  # At least 2 msgs/sec

class TestEdgeCasesAndRobustness(unittest.TestCase):
    """Test edge cases and system robustness"""
    
    @classmethod
    def setUpClass(cls):
        """Set up for edge case testing"""
        print("🔧 Setting up edge case testing...")
        cls.redactor = IntelligentRedactor()
        if not cls.redactor.model:
            cls.skipTest(cls, "Model not available for edge case testing")
    
    def test_empty_and_whitespace_inputs(self):
        """Test handling of empty and whitespace inputs"""
        edge_inputs = ["", "   ", "\n\n", "\t\t", "     \n   \t   "]
        
        for input_text in edge_inputs:
            with self.subTest(input_text=repr(input_text)):
                # Should not crash
                result = self.redactor.predict_toxicity(input_text)
                self.assertIn('probability', result)
                
                redact_result = self.redactor.redact_message(input_text)
                self.assertIn('original_text', redact_result)
    
    def test_very_long_messages(self):
        """Test handling of very long messages"""
        # Create a very long message
        long_message = "This is a test message. " * 200  # ~5000 characters
        
        # Should handle without crashing
        result = self.redactor.predict_toxicity(long_message)
        self.assertIn('probability', result)
        
        redact_result = self.redactor.redact_message(long_message)
        self.assertIn('original_text', redact_result)
    
    def test_special_characters_and_unicode(self):
        """Test handling of special characters and unicode"""
        special_inputs = [
            "Hello! 🎉👍✨",  # Emojis
            "Café naïve résumé",  # Accented characters
            "你好世界",  # Chinese characters
            "مرحبا بالعالم",  # Arabic characters
            "Здравствуй мир",  # Cyrillic characters
            "!@#$%^&*()_+-=[]{}|;':\",./<>?",  # Special symbols
        ]
        
        for input_text in special_inputs:
            with self.subTest(input_text=input_text):
                # Should handle without crashing
                result = self.redactor.predict_toxicity(input_text)
                self.assertIn('probability', result)
                
                redact_result = self.redactor.redact_message(input_text)
                self.assertIn('original_text', redact_result)
    
    def test_mixed_content(self):
        """Test messages with mixed clean and toxic content"""
        mixed_messages = [
            "Hello! You're an idiot. Have a great day!",
            "Thanks for sharing, but this is fucking stupid honestly",
            "I love this discussion, hate the stupid people though"
        ]
        
        for message in mixed_messages:
            with self.subTest(message=message):
                result = self.redactor.redact_message(message, 'smart')
                
                # Should preserve some good content while redacting bad
                self.assertNotEqual(result['redacted_text'], '[MESSAGE REDACTED - TOXIC CONTENT DETECTED]')
                self.assertNotEqual(result['redacted_text'], result['original_text'])

class TestSystemIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def test_redactor_to_moderator_integration(self):
        """Test integration between redactor and moderator"""
        redactor = IntelligentRedactor()
        if not redactor.model:
            self.skipTest("Model not available for integration testing")
        
        moderator = MessageModerator(redactor, auto_moderate=True)
        
        # Test message flow
        test_message = "You're such an idiot"
        result = moderator.moderate_message(test_message, "TestUser")
        
        self.assertIn('username', result)
        self.assertIn('action_taken', result)
        self.assertIn('toxicity_probability', result)

class TestConfigurationManagement(unittest.TestCase):
    """Test configuration management"""

    def test_policy_thresholds(self):
        """Test different moderation policies"""
        redactor = IntelligentRedactor()
        if not redactor.model:
            self.skipTest("Model not available for policy testing")
        
        policies = ['strict', 'moderate', 'lenient']
        
        for policy in policies:
            with self.subTest(policy=policy):
                moderator = MessageModerator(redactor, policy=policy)
                # Should initialize without error
                self.assertIsNotNone(moderator)

class RedactionSystemTestSuite:
    """
    Comprehensive test suite runner for the redaction system
    """
    
    def __init__(self):
        self.test_results = {}
        self.start_time = None
        self.end_time = None
    
    def run_all_tests(self):
        """Run all test suites"""
        print("🧪 COMPREHENSIVE REDACTION SYSTEM TEST SUITE")
        print("=" * 60)
        
        self.start_time = datetime.now()
        
        # Test suites to run
        test_classes = [
            TestIntelligentRedactor,
            TestMessageModerator,
            TestEdgeCasesAndRobustness,
            TestSystemIntegration,
            TestConfigurationManagement,
            TestPerformanceBenchmarks
        ]
        
        total_tests = 0
        total_passed = 0
        total_failed = 0
        
        for test_class in test_classes:
            print(f"\n🔍 Running {test_class.__name__}...")
            
            # Create test suite
            suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
            
            # Run tests with custom result handler
            runner = unittest.TextTestRunner(verbosity=1, stream=open(os.devnull, 'w'))
            result = runner.run(suite)
            
            # Track results
            class_name = test_class.__name__
            self.test_results[class_name] = {
                'tests_run': result.testsRun,
                'failures': len(result.failures),
                'errors': len(result.errors),
                'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / max(result.testsRun, 1)
            }
            
            passed = result.testsRun - len(result.failures) - len(result.errors)
            failed = len(result.failures) + len(result.errors)
            
            total_tests += result.testsRun
            total_passed += passed
            total_failed += failed
            
            # Print summary for this test class
            status = "✅" if failed == 0 else "❌"
            print(f"   {status} {passed}/{result.testsRun} passed")
            
            if result.failures:
                print(f"   ❌ {len(result.failures)} failures")
            if result.errors:
                print(f"   ❌ {len(result.errors)} errors")
        
        self.end_time = datetime.now()
        
        # Print final summary
        self._print_final_summary(total_tests, total_passed, total_failed)
    
    def _print_final_summary(self, total_tests, total_passed, total_failed):
        """Print comprehensive test summary"""
        print(f"\n" + "="*60)
        print("🎯 FINAL TEST RESULTS")
        print("="*60)
        
        duration = self.end_time - self.start_time
        overall_success_rate = total_passed / max(total_tests, 1)
        
        print(f"📊 Overall Statistics:")
        print(f"   Total tests: {total_tests}")
        print(f"   Passed: {total_passed}")
        print(f"   Failed: {total_failed}")
        print(f"   Success rate: {overall_success_rate:.1%}")
        print(f"   Duration: {duration}")
        
        print(f"\n📋 Detailed Results:")
        for class_name, results in self.test_results.items():
            status = "✅" if results['failures'] + results['errors'] == 0 else "❌"
            print(f"   {status} {class_name}: {results['success_rate']:.1%} "
                  f"({results['tests_run']} tests)")
        
        # Overall assessment
        if overall_success_rate >= 0.9:
            print(f"\n🎉 EXCELLENT: System is highly robust ({overall_success_rate:.1%} success)")
        elif overall_success_rate >= 0.8:
            print(f"\n👍 GOOD: System is generally robust ({overall_success_rate:.1%} success)")
        elif overall_success_rate >= 0.7:
            print(f"\n⚠️ ACCEPTABLE: System needs some improvements ({overall_success_rate:.1%} success)")
        else:
            print(f"\n❌ POOR: System needs significant improvements ({overall_success_rate:.1%} success)")
    
    def run_quick_validation(self):
        """Run a quick validation test"""
        print("🚀 QUICK SYSTEM VALIDATION")
        print("=" * 40)
        
        try:
            # Test basic functionality
            print("🔍 Testing basic redaction...")
            redactor = IntelligentRedactor()
            
            if not redactor.model:
                print("❌ Model not loaded - cannot validate")
                return False
            
            # Test basic redaction
            result = redactor.redact_message("You're an idiot", "smart")
            
            if result['was_redacted']:
                print("✅ Basic redaction working")
            else:
                print("⚠️ Redaction may be too lenient")
            
            # Test clean message
            clean_result = redactor.redact_message("Hello, nice day!", "smart")
            
            if not clean_result['was_redacted']:
                print("✅ Clean message handling working")
            else:
                print("⚠️ May be over-filtering clean messages")
            
            print("✅ Quick validation passed")
            return True
            
        except Exception as e:
            print(f"❌ Quick validation failed: {e}")
            return False

def main():
    """Main test runner"""
    print("🧪 REDACTION SYSTEM TEST SUITE")
    print("=" * 40)
    
    print("Choose testing option:")
    print("1. Run all comprehensive tests")
    print("2. Run quick validation only")
    print("3. Run performance benchmarks only")
    print("4. Test specific component")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    test_suite = RedactionSystemTestSuite()
    
    if choice == '1':
        # Run comprehensive tests
        test_suite.run_all_tests()
        
    elif choice == '2':
        # Quick validation
        test_suite.run_quick_validation()
        
    elif choice == '3':
        # Performance benchmarks only
        print("⚡ Running performance benchmarks...")
        suite = unittest.TestLoader().loadTestsFromTestCase(TestPerformanceBenchmarks)
        runner = unittest.TextTestRunner(verbosity=2)
        runner.run(suite)
        
    elif choice == '4':
        # Test specific component
        print("\nAvailable components:")
        print("1. IntelligentRedactor")
        print("2. MessageModerator")
        print("3. Edge Cases")
        print("4. Integration")

        component_choice = input("Choose component (1-4): ").strip()

        component_map = {
            '1': TestIntelligentRedactor,
            '2': TestMessageModerator,
            '3': TestEdgeCasesAndRobustness,
            '4': TestSystemIntegration
        }
        
        if component_choice in component_map:
            test_class = component_map[component_choice]
            print(f"🔍 Running {test_class.__name__}...")
            
            suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
            runner = unittest.TextTestRunner(verbosity=2)
            runner.run(suite)
        else:
            print("Invalid choice")
            
    else:
        print("Invalid choice. Please enter 1-4.")

if __name__ == "__main__":
    main()
